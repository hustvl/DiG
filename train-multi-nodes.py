# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import math
import torch
import torchvision
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import torchvision.transforms.functional as F

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models_dit import DiT_models
from models_dig import DiG_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# import amp
from torch.amp import autocast as amp_autocast
from torch.cuda.amp import GradScaler



#################################################################################
#                             Training Helper Functions                         #
#################################################################################
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        device = int(os.environ['LOCAL_RANK'])
        print(f"LOCAL_RANK, RANK and WORLD_SIZE in environ: {device}/{rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(device)

    # Setup DDP:
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()


    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    if "DiG" in args.model:
        target_model = DiG_models
    elif "DiT" in args.model:
        target_model = DiT_models
    else:
        print("error")

    amp_enable = False
    scaler = None
    if args.amp == "fp16":
        dtype = torch.float16
        amp_enable = True
        scaler = GradScaler()
    elif args.amp == "bf16":
        dtype = torch.bfloat16
        amp_enable = True
        scaler = GradScaler()
    elif args.amp == "fp32":
        dtype = torch.float32
    else:
        assert "error dtype"

    if args.data_type == "cifar10" or args.data_type == "celeba":
        model = target_model[args.model](
            input_size=args.image_size,
            num_classes=args.num_classes,
            in_channels=3
        )
    else:
        model = target_model[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).cuda()  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    model = DDP(model.cuda())#.to(dtype))
    model_without_ddp = model.module
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule



    vae = AutoencoderKL.from_pretrained(f"/mnt/bn/lianghuidata/ckpts/sd-vae-ft-{args.vae}").cuda()#.to(dtype)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("DiT Arch: ", model_without_ddp)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.data_type == "in1k":
        dataset = ImageFolder(args.data_path, transform=transform)
    elif args.data_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root="/mnt/bn/lianghuidata/datasets/cifar",
            train=True,
            download=False,
            transform=transform,
        )
    elif args.data_type == "celeba":
        # define transform
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        transform = transforms.Compose([Crop(x1, x2, y1, y2), transforms.Resize(args.image_size),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        dataset = torchvision.datasets.CelebA(root='/mnt/bn/lianghuidata/datasets', split="train", transform=transform, download=False)
    else:
        assert "error data type"
        
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Batchsize per GPU {int(args.global_batch_size // dist.get_world_size())}")
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        ema.load_state_dict(checkpoint['ema'])

        # load ema
        # import io
        # mem_file = io.BytesIO()
        # torch.save({'state_dict_ema':checkpoint['ema']}, mem_file)
        # mem_file.seek(0)
        # ema._load_checkpoint(mem_file)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.cuda()

            if args.num_classes > 0:
                y = y.cuda()
            else:
                y = None
            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x = vae.encode(x).latent_dist.sample().mul_(0.18215)


            if args.data_type != "cifar10" and args.data_type != "celeba":
                # VAE encode
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    # x = x.to(dtype)
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    
                    # cast back to fp32 for bettet diffusion accuracy
                    # x = x.to(torch.float32)

            with amp_autocast(device_type="cuda", enabled=amp_enable, dtype=dtype):
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],)).cuda()
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            loss = loss_dict["loss"].mean()

            # loss_value = loss.item()
            # if math.isfinite(loss_value) or math.isnan(loss_value): 
            #     print("nan or inf")
            #     continue

            opt.zero_grad()

            if amp_enable and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if amp_enable and scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps).cuda()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()) + list(DiG_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1_000)
    parser.add_argument("--amp", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-type", type=str, default="in1k", choices=["in1k", "cifar10", "celeba"])
    args = parser.parse_args()
    main(args)
