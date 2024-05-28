# -*- coding: utf-8 -*-

# "Gated Linear Attention Transformers with Hardware-Efficient Training"[https://arxiv.org/abs/2312.06635]

from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from transformers.activations import ACT2FN
try:
    from fla.modules.activations import ACT2FN
except:
    from transformers.activations import ACT2FN


from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.gla.naive import naive_recurrent_gla

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fla.models.gla.configuration_gla import GLAConfig
from fla.modules import FusedCrossEntropyLoss, RMSNorm, ShortConvolution, FusedRMSNormSwishGate
from fla.modules.activations import swiglu_linear

logger = logging.get_logger(__name__)


class GatedLinearAttention(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa

    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        share_conv_kernel (bool, Optional):
            Whether to apply convolutions berfore q/k/v mapping, only taking effects when `use_short_conv`. Default: `True`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        elementwise_affine (bool, Optional):
            If `True`, applies elementwise affine to LayerNorm with learnable parameters. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        clamp_min (float, Optional):
            The minimum value for the gate logits. Default: None.
        fuse_norm (bool, Optional):
            Whether to fuse the norm and the output gate for better memory footprint. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        layernorm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
        *args, **kwargs
    ) -> GatedLinearAttention:
        super().__init__()

        # rename
        hidden_size = d_model
        norm_eps = layernorm_eps

        # default initialization
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk', 'naive'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim, bias=True))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            # todo: fix this
            try:
                self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            except:
                self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(self.head_v_dim, elementwise_affine, norm_eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.use_short_conv:
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
            else:
                conv_state_q = last_state[0] if use_cache else None
                conv_state_k = last_state[1] if use_cache else None
                conv_state_v = last_state[2] if use_cache else None
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                q = self.q_conv1d(q, attention_mask, conv_state_q)
                k = self.k_conv1d(k, attention_mask, conv_state_k)
                v = self.v_conv1d(v, attention_mask, conv_state_v)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))
        gk = rearrange(self.gk_proj(hidden_states), 'b n (h d) -> b h n d', h=self.num_heads)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        recurrent_state = last_state[-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
            pass
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'naive':
            o, recurrent_state = naive_recurrent_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    last_state = (conv_state, recurrent_state)
                else:
                    last_state = (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = rearrange(o, 'b h l d -> b l h d')
        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b l h d -> b l (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
        o = self.o_proj(o)

        return o

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size


class GLAMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> GLAMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)


class GLABlock(nn.Module):
    def __init__(self, config: GLAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GatedLinearAttention(
            d_model=config.hidden_size,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_heads=config.num_heads,
            gate_fn=config.hidden_act,
            layernorm_eps=config.rms_norm_eps,
            mode=config.attn_mode,
            clamp_min=config.clamp_min,
            fuse_norm=config.fuse_norm,
            layer_idx=layer_idx,
            if_norm_qkv=config.if_norm_qkv,
            if_scale_qkv=config.if_scale_qkv,
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GLAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        # currently not supported
        attn_weights, present_key_value = None, None

        hidden_states = self.attn_norm(hidden_states)

        hidden_states, attentions, past_key_values = self.attn(hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            past_key_value=past_key_value, 
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # outputs = (hidden_states, attentions, past_key_values)

        return hidden_states


class GLAPreTrainedModel(PreTrainedModel):
    config_class = GLAConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GLAModel(GLAPreTrainedModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`GLAModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            _, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            _, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GLAForCausalLM(GLAPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



if __name__ == '__main__':
    batch = 4
    seq_len = 1024

    d_model = 2048
    # x = torch.randn(batch, seq_len, d_model).to(torch.bfloat16).cuda().requires_grad_(True)
    # model = GatedLinearAttention(use_gk=True, use_gv=True, mode='fused_chunk').to(torch.bfloat16).cuda()
    # y = model(x)
    # print(y.shape)
    # y.sum().backward()
    # print(x.grad.shape)

    for act in ['swish']:
        org = GatedLinearAttention(d_model=d_model, gate_fn=act, fuse_norm=False).to(torch.bfloat16).cuda()
        fused = GatedLinearAttention(d_model=d_model, gate_fn=act, fuse_norm=True).to(torch.bfloat16).cuda()
        fused.q_proj.weight.data.copy_(org.q_proj.weight.data)
        fused.k_proj.weight.data.copy_(org.k_proj.weight.data)
        fused.v_proj.weight.data.copy_(org.v_proj.weight.data)
        fused.g_proj.weight.data.copy_(org.g_proj.weight.data)
        fused.o_proj.weight.data.copy_(org.o_proj.weight.data)
        fused.gk_proj[0].weight.data.copy_(org.gk_proj[0].weight.data)
        fused.gk_proj[1].weight.data.copy_(org.gk_proj[1].weight.data)
        fused.gk_proj[1].bias.data.copy_(org.gk_proj[1].bias.data)

        x = torch.randn(batch, seq_len, d_model).to(torch.bfloat16).cuda()
        org_x = x.clone().requires_grad_(True)
        fused_x = x.clone().requires_grad_(True)
        org_o = org(org_x)
        fused_o = fused(fused_x)
        org_o.sum().backward()
        fused_o.sum().backward()
        breakpoint()
        assert org_o.allclose(fused_o, 0, 1e-2), "output not equal"
        assert org_x.grad.allclose(fused_x.grad, 0, 1e-2), "grad not equal"
