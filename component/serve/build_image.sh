#!/bin/bash

image_name=10.161.31.82:5000/phm
image_tag=0.1-gnn-serve
full_image_name=${image_name}:${image_tag}
cd "$(dirname "$0")"

docker build -t "${full_image_name}" .
docker push ${full_image_name}
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"