#!/bin/bash

arch=$1
num=$2
resume=$3

python run_eval.py \
    --num_classes 200 \
    --num_descriptive "$num" \
    --num_prototypes 202 \
    --arch "$arch" \
    --pretrained \
    --proto_depth 256 \
    --prototype_activation_function log \
    --last_layer \
    --use_thresh \
    --resume "$resume"