#!/bin/bash
base_dir=$(pwd)
components_dir=$base_dir/component

for c in $components_dir/*/; do
    cd $c && ./build_image.sh
done

dsl-compile --py $base_dir/pipeline/pipeline.py --output $base_dir/gnn-pipeline.tar.gz
