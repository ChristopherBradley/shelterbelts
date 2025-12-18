#!/bin/bash

# year=2020
# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_${year}_actnsw_4326_weightings/subfolders"

# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/"
# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/subfolders"
BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2024/subfolders"


for folder in "$BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Submitting job for $folder"
        qsub -v base_dir="$folder" merge_lidar.pbs
    fi
done