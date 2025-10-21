#!/bin/bash

# Driver script to submit merge jobs for each Sentinel subfolder
BASE_SENTINEL_DIR="/scratch/xe2/cb8590/Nick_sentinel/tiles_todo"

# We had 7k leftover after the first run, then 178 leftover after then second run.

for folder in "$BASE_SENTINEL_DIR"/subfolder*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        echo "Submitting job for $folder_name"
        qsub -v SENTINEL_FOLDER="$folder",STUB="$folder_name" merge_inputs_outputs.pbs
    fi
done
