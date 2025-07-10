#!/bin/bash

for option in 0 1 2; do
    sbatch train_2gpus.sh $option
    sbatch infer_2gpus.sh $option
done