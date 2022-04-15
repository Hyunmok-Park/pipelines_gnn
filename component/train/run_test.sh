#!/bin/bash

image_name=10.161.31.82:5000/phm
image_tag=0.1-gnn-train
full_image_name=${image_name}:${image_tag}

docker run --rm --gpus=all --ipc=host ${full_image_name}
docker ps -a | grep tf
