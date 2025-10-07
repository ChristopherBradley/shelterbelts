#!/bin/bash

# Remove tiff files that have already been processed correctly. This is fine to do in scratch because I've saved a complete copy in gdata.
TIFF_DIR="/scratch/xe2/cb8590/Nick_Aus_treecover_10m"
CSV_DIR="/scratch/xe2/cb8590/alphaearth"

for tiff in "$TIFF_DIR"/*.tiff; do
    base=$(basename "$tiff" .tiff)   # e.g. g1_0120_binary_tree_cover_10m
    csv="$CSV_DIR/${base}_alpha_earth_embeddings.csv"
    
    if [[ -f "$csv" ]]; then
        rm "$tiff"
    fi
done

