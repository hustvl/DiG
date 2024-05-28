# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from fla.models import GLAConfig
from module_dig import GLABlock as SeqGLABlock

from models_dit import TimestepEmbedder, LabelEmbedder, modulate, FinalLayer


#################################################################################
#                                 Core DiG Model                                #
#################################################################################

class SeqDiGBlock(SeqGLABlock):
    """
    A DiG block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, config, layer_idx, drop_path=0, if_dw_conv=False, conv_kernel_size=3, if_in_block_conv=False, if_rms_norm=False, if_skip=False, **block_kwargs):
        super().__init__(config, layer_idx)
        
        if if_rms_norm:
            self.norm1 = self.attn_norm
            self.norm2 = self.mlp_norm
        else:
            del self.attn_norm
            del self.mlp_norm
            self.norm1 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

        self.if_dw_conv = if_dw_conv
        self.if_in_block_conv = if_in_block_conv
        if self.if_dw_conv:
            self.dw_conv = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=conv_kernel_size, stride=1, padding=conv_kernel_size//2, groups=config.hidden_size)
            self.hidden_size = config.hidden_size
            self.block_kwargs = block_kwargs

        self.layer_idx = layer_idx

        self.if_skip = if_skip
        if self.if_skip:
            self.skip_linear = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, c, skip=None):

        if self.if_skip and skip is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        if self.if_dw_conv and self.if_in_block_conv:
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.attn(
                self.dw_conv(
                    modulate(
                        self.norm1(hidden_states), shift_msa, scale_msa
                    ).transpose(1, 2).reshape(-1, self.hidden_size, self.block_kwargs['edge_len'], self.block_kwargs['edge_len'])
                ).flatten(2).transpose(1, 2)
            )
        else:
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(hidden_states), shift_msa, scale_msa))
        
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(hidden_states), shift_mlp, scale_mlp))

        if self.if_dw_conv and not self.if_in_block_conv:
            hidden_states = hidden_states.transpose(1, 2).reshape(-1, self.hidden_size, self.block_kwargs['edge_len'], self.block_kwargs['edge_len'])
            hidden_states = self.dw_conv(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states

class DiG(nn.Module):
    """
    Diffusion model with a GLA backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        enable_flashattn: bool = False,
        enable_layernorm_kernel: bool = False,
        enable_modulate_kernel: bool = False,
        sequence_parallel_size: int = 1,
        sequence_parallel_group: Optional[ProcessGroup] = None,
        sequence_parallel_type: str = None,
        dtype: torch.dtype = torch.float32,
        use_video: bool = False,
        text_encoder: str = None,
        attn_model="fused_chunk",
        expand_k=0.5,
        expand_v=1,
        hidden_act="swish",
        bid_mode='layer',
        use_dirpe=False,
        if_dw_conv=False,
        if_quad_dir=False,
        conv_kernel_size=3,
        if_in_block_conv=False,
        if_rms_norm=False,
        if_norm_qkv=False,
        if_scale_qkv=False,
        if_old=False,
        if_skip=False,
        if_zero_init_skip=False,
        if_fuse_norm=True,
        **kwargs,
    ):
        super().__init__()

        if bid_mode == 'seq' or bid_mode == "layer":
            DiGBlock = SeqDiGBlock 
        else:
            raise NotImplementedError

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.sequence_parallel_size = sequence_parallel_size
        self.sequence_parallel_group = sequence_parallel_group
        self.sequence_parallel_type = sequence_parallel_type
        self.if_quad_dir = if_quad_dir

        config = GLAConfig(
            hidden_size=hidden_size,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            num_heads=num_heads,
            attn_mode=attn_model,
            expand_k=expand_k,
            expand_v=expand_v,
            hidden_act=hidden_act,
            bid_mode=bid_mode,
            use_dirpe=use_dirpe,
            rms_norm_eps=1e-6,
            if_norm_qkv=if_norm_qkv,
            if_scale_qkv=if_scale_qkv,
            fuse_norm=if_fuse_norm,
        )
        self.config = config

        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiGBlock(config, layer_idx, if_dw_conv=if_dw_conv, conv_kernel_size=conv_kernel_size, if_in_block_conv=if_in_block_conv, if_rms_norm=if_rms_norm, edge_len=int(self.x_embedder.num_patches ** 0.5), if_skip=(if_skip and layer_idx > depth // 2)) for layer_idx in range(depth)
        ])

        if self.num_classes > 0:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        else:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, condition=False)

        self.if_skip = if_skip
        self.if_zero_init_skip = if_zero_init_skip
            
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) and not getattr(module, "_is_hf_initialized", False):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv2d) and module.weight.shape[-1] >= 3:
                kernel_size = module.weight.size(2)
                if kernel_size % 2 == 1:
                    with torch.no_grad():
                        module.weight.fill_(0)
                        center = kernel_size // 2
                        module.weight[:, :, center, center] = 1

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.num_classes > 0:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiG blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.num_classes > 0:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.if_zero_init_skip:
            for block in self.blocks:
                if block.if_skip:
                    with torch.no_grad():
                        nn.init.constant_(block.skip_linear.weight, 0)
                        cout, cin = block.skip_linear.weight.shape
                        for i in range(cout):
                            block.skip_linear.weight[i, i] = 1.
                        nn.init.constant_(block.skip_linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiG.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)

        if y is not None:
            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                                # (N, D)
        else:
            c = t

        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x, c, skip=skips.pop() if i>len(self.blocks)//2 and self.if_skip else None)                      # (N, T, D)

            if self.config.bid_mode == "layer":
                x = x.flip(1)

            if i % 2 == 1 and self.if_quad_dir:
                x = x.reshape(x.shape[0], int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5), x.shape[2]).transpose(1, 2).flatten(1, 2)

            if i < len(self.blocks) // 2 - 1 and self.if_skip:
                skips.append(x)

        x = self.final_layer(x, c if self.num_classes > 0 else None)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiG Configs                                  #
#################################################################################


def DiG_XL_2_bid_layer_dwconv_qdir_head18(**kwargs):
    return DiG(depth=28, hidden_size=1152, patch_size=2, num_heads=18, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, **kwargs)

def DiG_XL_2_bid_layer_dwconv_qdir_head18_skip(**kwargs):
    return DiG(depth=28, hidden_size=1152, patch_size=2, num_heads=18, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, if_skip=True, **kwargs)

def DiG_XL_2_bid_layer_dwconv_qdir_head18_skipzero(**kwargs):
    return DiG(depth=28, hidden_size=1152, patch_size=2, num_heads=18, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, if_skip=True, if_zero_init_skip=True, **kwargs)

def DiG_L_2_bid_layer_dwconv_qdir(**kwargs):
    return DiG(depth=24, hidden_size=1024, patch_size=2, num_heads=16, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, **kwargs)

def DiG_B_2_bid_layer_dwconv_qdir(**kwargs):
    return DiG(depth=12, hidden_size=768, patch_size=2, num_heads=12, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, **kwargs)

def DiG_S_2_bid_layer_dwconv_qdir(**kwargs):
    return DiG(depth=12, hidden_size=384, patch_size=2, num_heads=6, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, **kwargs)

def DiG_S_2_bid_layer_dwconv_qdir_skip(**kwargs):
    return DiG(depth=12, hidden_size=384, patch_size=2, num_heads=6, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, if_skip=True, **kwargs)

def DiG_S_2_bid_layer_dwconv_qdir_skipzero(**kwargs):
    return DiG(depth=12, hidden_size=384, patch_size=2, num_heads=6, bid_mode='layer', if_dw_conv=True, if_quad_dir=True, if_skip=True, if_zero_init_skip=True, **kwargs)


DiG_models = {
    # XL mdoels
    'DiG-XL/2-bid-layer-dwconv-qdir-head18': DiG_XL_2_bid_layer_dwconv_qdir_head18,
    'DiG-XL/2-bid-layer-dwconv-qdir-head18-skip': DiG_XL_2_bid_layer_dwconv_qdir_head18_skip,
    'DiG-XL/2-bid-layer-dwconv-qdir-head18-skipzero': DiG_XL_2_bid_layer_dwconv_qdir_head18_skipzero,
    
    # L models
    'DiG-L/2-bid-layer-dwconv-qdir': DiG_L_2_bid_layer_dwconv_qdir,

    # B models
    'DiG-B/2-bid-layer-dwconv-qdir': DiG_B_2_bid_layer_dwconv_qdir,

    # S models
    'DiG-S/2-bid-layer-dwconv-qdir': DiG_S_2_bid_layer_dwconv_qdir,
    'DiG-S/2-bid-layer-dwconv-qdir-skip': DiG_S_2_bid_layer_dwconv_qdir_skip,
    'DiG-S/2-bid-layer-dwconv-qdir-skipzero': DiG_S_2_bid_layer_dwconv_qdir_skipzero,
}
