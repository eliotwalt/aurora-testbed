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

srun torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train.py --num_epochs=500 --bf16 \
        --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction \
        --no_ddp