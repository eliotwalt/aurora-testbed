#!/bin/bash
#SBATCH --job-name=aurora-infer
#SBATCH --partition=gpu_h100
#SBATCH --time=1:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/infer/2gpus/%j.log
#SBATCH --error=logs/infer/2gpus/%j.log

export CUDA_LAUNCH_BLOCKING=1

source env/venv_h100/bin/activate
echo "Environment Info:"
echo " * Torchrun path: $(which torchrun)"
echo " * Python path: $(which python)"
echo " * Python version: $(python --version)"
echo " * pytorch version: $(python -c 'import torch; print(torch.__version__)')"

srun torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    infer.py --num_steps=196 --bf16