#!/bin/bash


IF_INSTALL_ENV=${1:-env}

DATA_PATH="/mnt/bn/lianghuidata/datasets/ImageNet/train"
# path setting
if [ "$IF_INSTALL_ENV" = env ] ; then
    echo "installing env..."
    
    pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    pip3 install triton

    pip3 install diffusers
    pip3 install tensorboard
    pip3 install timm
    pip3 install transformers
    pip3 install accelerate

    pip3 install fvcore

    git clone https://github.com/sustcsonglin/flash-linear-attention
    git checkout 36743f3f14e47f23c1ad45cf5de727dbacb5600e
    cd flash-linear-attention
    pip3 install -e .

    pip3 install opt_einsum

    pip3 install torchdiffeq

    pip3 install ftfy
    pip3 install PyAV

    pip3 uninstall -y timm
    pip3 install timm
fi

# cd work_dir
cd DiG_code

MODEL="DiG-S/2-bid-layer-dwconv-qdir"
APPENDIX="-in1k-size256-bs256-1node-local-py-fix-a100"

MODELALIAS=${MODEL//\//d}

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port 23112 --use_env train-multi-nodes.py \
    --model $MODEL \
    --global-batch-size 256 \
    --global-seed 0 \
    --vae ema \
    --num-workers 32 \
    --num-classes 1000 \
    --image-size 256 \
    --epochs 1400 \
    --results-dir results/$MODELALIAS$APPENDIX \
    --log-every 100 \
    --ckpt-every 1000 \
    --amp fp32 \
    --lr 1e-4 \
    --data-path $DATA_PATH 
