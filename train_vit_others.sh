#!/bin/bash

arch=$1
num=$2
data=$3
num_classes=$4

python main.py \
    --data_type "$data" \
    --num_classes "$num_classes" \
    --batch_size 80 \
    --lr 0.001 \
    --epochs 40 \
    --num_descriptive "$num" \
    --num_prototypes 202 \
    --results ./outputs \
    --earlyStopping 8 \
    --use_scheduler \
    --arch "$arch" \
    --pretrained \
    --proto_depth 256 \
    --warmup_time 15 \
    --warmup \
    --prototype_activation_function log \
    --top_n_weight 0 \
    --last_layer \
    --use_thresh \
    --mixup_data \
    --pp_ortho \
    --pp_gumbel \
    --gumbel_time 15 \
    --data_train ./datasets/ \
    --data_push ./datasets/ \
    --data_test ./datasets/
