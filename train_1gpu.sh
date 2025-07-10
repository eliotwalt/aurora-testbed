#!/bin/bash
#SBATCH --job-name=aurora-train
#SBATCH --partition=gpu_h100
#SBATCH --time=1:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train/%j.log
#SBATCH --error=logs/train/%j.log

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
        echo "|$option|$SLURM_JOB_ID|500|true|false|false|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --small \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    1)
        # small bf16, no autocast, all checkpointing
        echo "|$option|$SLURM_JOB_ID|500|true|true|false|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --small --bf16 \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    2)
        # small no bf16, autocast, all checkpointing
        echo "|$option|$SLURM_JOB_ID|500|true|false|true|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --small --autocast \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    3)
        # small bf16, autocast, all checkpointing
        echo "|$option|$SLURM_JOB_ID|500|true|true|true|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --small --bf16 --autocast \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    4)
        # small no bf16, autocast, Basic3DEncoderLayer Basic3DDecoderLayer checkpointing
        echo "|$option|$SLURM_JOB_ID|500|true|false|true|Basic3DEncoderLayer Basic3DDecoderLayer|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --small --autocast \
            --checkpointing_module_names Basic3DEncoderLayer Basic3DDecoderLayer
        ;;
    5)
        # large no bf16, no autocast, all checkpointing
        echo "|$option|$SLURM_JOB_ID|500|false|false|false|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    6)
        # large bf16, no autocast, all checkpointing
        echo "|$option|$SLURM_JOB_ID|500|false|true|false|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --bf16 \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    7)
        # large no bf16, autocast, all checkpointing
        echo "|$option|$SLURM_JOB_ID|500|false|false|true|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --autocast \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    8)
        # large bf16, autocast, all checkpointing
        echo "|$option|$SLURM_JOB_ID|500|false|true|true|all|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --bf16 --autocast \
            --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    9)
        # large no bf16, autocast, Basic3DEncoderLayer Basic3DDecoder
        echo "|$option|$SLURM_JOB_ID|500|false|false|true|Basic3DEncoderLayer Basic3DDecoderLayer|"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py \
            --num_epochs=500 --autocast \
            --checkpointing_module_names Basic3DEncoderLayer Basic3DDecoderLayer
        ;;
    *)
        echo "Invalid option: $option"
        exit 1
        ;;
esac