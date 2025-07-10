#!/bin/bash

train_1gpu.sh 1
train_1gpu.sh 2
train_1gpu.sh 3
train_1gpu.sh 4
train_1gpu.sh 5
train_1gpu.sh 6

infer_1gpu.sh 1
infer_1gpu.sh 2
infer_1gpu.sh 3
infer_1gpu.sh 4
infer_1gpu.sh 5
infer_1gpu.sh 6