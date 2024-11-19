#!/bin/bash

# bash train_vit.sh dinov2_vits_exp 10
# bash train_vit.sh dinov2_vits_exp 5

# bash train_vit.sh dinov2_vitb_exp 10
# bash train_vit.sh dinov2_vitb_exp 5

bash train_vit_others.sh dinov2_vitb_exp 5 cars 196
bash train_vit_others.sh dinov2_vitb_exp 3 cars 196
bash train_vit_others.sh dinov2_vits_exp 5 cars 196
bash train_vit_others.sh dinov2_vits_exp 3 cars 196