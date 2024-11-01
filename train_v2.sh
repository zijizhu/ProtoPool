#!/bin/bash

bash train_vit.sh dinov2_vits_exp 3

bash train_vit.sh dino_vitb16 10
bash train_vit.sh dino_vitb16 5
bash train_vit.sh dino_vitb16 3

bash train_vit.sh dinov2_vitb_exp 3
bash train_vit.sh dinov2_vitb_exp 5
bash train_vit.sh dinov2_vitb_exp 10
