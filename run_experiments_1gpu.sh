#!/bin/bash

for i in {0..9}
do
    sbatch train_1gpu.sh $i
    sbatch infer_1gpu.sh $i
done