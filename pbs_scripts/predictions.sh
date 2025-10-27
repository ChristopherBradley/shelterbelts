#!/bin/bash

# Directory containing the split gpkg files
GPKG_DIR="/g/data/xe2/cb8590/Outlines/BARRA_bboxs/BARRA_bboxs_nsw"
OUTDIR="/scratch/xe2/cb8590/barra_trees_s4_2020"

# Loop through each gpkg and submit a PBS job
for gpkg in "$GPKG_DIR"/BARRA_bboxs_nsw_*.gpkg; do
    fname=$(basename "$gpkg" .gpkg)
    qsub -N "predictions" \
         -v GPKG="$gpkg",OUTDIR="$OUTDIR" \
         predictions.pbs
done