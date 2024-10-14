#!/bin/bash

set -x

bash train.sh dinov2_vits_exp 10
bash train.sh dinov2_vits_exp 20

bash train.sh dinov2_vitb_exp 10
bash train.sh dinov2_vitb_exp 20