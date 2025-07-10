#!/bin/bash
for option in 0 1 2; do
    sbatch train_1gpu_no_ddp.sh $option
done