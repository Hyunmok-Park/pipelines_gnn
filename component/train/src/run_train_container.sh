#!/bin/sh

python run_exp_local.py --train_data=$1 --val_data=$2
#bentoml serve simple_gnn:latest --production

