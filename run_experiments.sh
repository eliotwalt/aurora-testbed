#!/bin/bash

sbatch train_1gpu.sh 1
sbatch train_1gpu.sh 2
sbatch train_1gpu.sh 3
sbatch train_1gpu.sh 4
sbatch train_1gpu.sh 5
sbatch train_1gpu.sh 6

sbatch infer_1gpu.sh 1
sbatch infer_1gpu.sh 2
sbatch infer_1gpu.sh 3
sbatch infer_1gpu.sh 4
sbatch infer_1gpu.sh 5
sbatch infer_1gpu.sh 6