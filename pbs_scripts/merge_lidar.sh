#!/bin/bash

BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders"

for folder in "$BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Submitting job for $folder"
        qsub -v base_dir="$folder" merge_lidar.pbs
    fi
done
