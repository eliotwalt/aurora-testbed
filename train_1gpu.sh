#!/bin/bash
#SBATCH --job-name=aurora-ddp
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16

source ../Xaurora/env/venv_h100/bin/activate

option=$1

case "$option" in
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
        # large bf16, no autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --bf16 \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    5)
        # large no bf16, autocast, all checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --autocast \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction  
        ;;
    6)
        # large no bf16, autocast, Basic3D checkpointing
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=10 --autocast \
            --checkpointing_module_names Basic3DEncoderLayer Basic3DDecoderLayer 
        ;;
    *)
        echo "Usage: $0 {1|2|3|4|5|6}"
        exit 1
        ;;
esac
