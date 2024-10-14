#!/bin/bash

arch=$1
num=$2

python main.py \
    --data_type birds \
    --num_classes 200 \
    --batch_size 80 \
    --lr 1e-3 \
    --epochs 10 \
    --num_descriptive "$num" \
    --num_prototypes 202 \
    --results ./outputs \
    --earlyStopping 10 \
    --use_scheduler \
    --arch "$arch" \
    --pretrained \
    --proto_depth 256 \
    --warmup_time 4 \
    --warmup \
    --prototype_activation_function log \
    --top_n_weight 0 \
    --last_layer \
    --use_thresh \
    --mixup_data \
    --pp_ortho \
    --pp_gumbel \
    --gumbel_time 30 \
    --data_train ./datasets/cub200_cropped/train_cropped_augmented/ \
    --data_push ./datasets/cub200_cropped/train_cropped/ \
    --data_test ./datasets/cub200_cropped/test_cropped/
