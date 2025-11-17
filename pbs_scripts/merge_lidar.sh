#!/bin/bash

# year=2020
# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_${year}_actnsw_4326_weightings/subfolders"

BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/"

for folder in "$BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Submitting job for $folder"
        qsub -v base_dir="$folder" merge_lidar.pbs
    fi
done


# Once finished I moved them into a new folder like this: mv /scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/*_merged_predicted.tif /scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted