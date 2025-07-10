#!/bin/bash
#SBATCH --job-name=env-h100
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python3.11 -m venv env/venv_h100
source env/venv_h100/bin/activate

pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 microsoft-aurora==1.7.0 timm==1.0.15