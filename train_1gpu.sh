#!/bin/bash
#SBATCH --job-name=aurora-ddp
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16

export CUDA_LAUNCH_BLOCKING=1

source env/venv_h100/bin/activate
echo "Environment Info:"
echo " * Torchrun path: $(which torchrun)"
echo " * Python path: $(which python)"
echo " * Python version: $(python --version)"
echo " * pytorch version: $(python -c 'import torch; print(torch.__version__)')"

option=$1
case "$option" in
    0) 
        # small no bf16, no autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --small \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
    1)
        # small bf16, no autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --small --bf16 \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    2)
        # small no bf16, autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --small --autocast \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    3)
        # small no bf16, autocast, Basic3D checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --small --autocast \
            --checkpointing_module_names Basic3DEncoderLayer Basic3DDecoderLayer
        ;;
    4)
        # large no bf16, no autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    5)
        # large bf16, no autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --bf16 \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    6)
        # large no bf16, autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --autocast \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    7)
        # large no bf16, autocast, Basic3D checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --autocast \
            --checkpointing_module_names Basic3DEncoderLayer Basic3DDecoderLayer
        ;;
    *)
        echo "Usage: $0 {0|1|2|3|4|5|6|7}"
        exit 1
        ;;
esac