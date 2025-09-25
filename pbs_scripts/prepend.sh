#!/bin/bash

# Prepend the folder name to each of the files
for f in /scratch/xe2/cb8590/lidar/*/uint8_percentcover_res10_height2m/uint8_percentcover_res10_height2m_footprints.gpkg; do
    folder=$(basename "$(dirname "$(dirname "$f")")")   # gets the * folder, e.g. DATA_12345
    dir=$(dirname "$f")                                # parent directory
    base=$(basename "$f")                              # original filename
    mv "$f" "$dir/${folder}_${base}"
done
