#!/bin/bash
#SBATCH --job-name=aurora-train
#SBATCH --partition=gpu_h100
#SBATCH --time=1:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train/1gpu_no_ddp/%j.log
#SBATCH --error=logs/train/1gpu_no_ddp/%j.log

export CUDA_LAUNCH_BLOCKING=1

source env/venv_h100/bin/activate
echo "Environment Info:"
echo " * Torchrun path: $(which torchrun)"
echo " * Python path: $(which python)"
echo " * Python version: $(python --version)"
echo " * pytorch version: $(python -c 'import torch; print(torch.__version__)')"

option=$1
case $option in
    0)
        # bf16 mode
        echo "$SLURM_JOB_ID: Running in bf16 mode"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py --num_epochs=500 --bf16 \
                --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    1)
        # autocast mode
        echo "$SLURM_JOB_ID: Running in autocast mode"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py --num_epochs=500 --autocast \
                --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    2)
        # none
        echo "$SLURM_JOB_ID: Running in FP32 mode"
        srun torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            train.py --num_epochs=500 \
                --checkpointing_module_names Perceiver3DEncoder Swin3DTransformerBackbone Basic3DEncoderLayer Basic3DDecoderLayer Perceiver3DDecoder LinearPatchReconstruction
        ;;
    *)
        echo "Invalid option: ${option}"
        exit 1
        ;;
esac