#!/bin/bash

# year=2020
# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_${year}_actnsw_4326_weightings/subfolders"

# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/"
# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/subfolders"
# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2024/subfolders"  # Used this one to merge the predictions
# BASE_DIR="/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2017/subfolders"  # Used this one to merge the predictions
# FOLDER_SUFFIX=

BASE_DIR=/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2024/expanded  # Used this one to merge the indices
# # FOLDER_SUFFIX=less_percentmethod
FOLDER_SUFFIX=default_percentmethod

for folder in "$BASE_DIR"/*"$FOLDER_SUFFIX"; do
    if [ -d "$folder" ]; then
        echo "Submitting job for $folder"
        qsub -v base_dir="$folder" merge_lidar.pbs
    fi
done