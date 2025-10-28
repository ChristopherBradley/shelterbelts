#!/bin/bash

# Directory containing the split gpkg files
GPKG_DIR="/g/data/xe2/cb8590/Outlines/BARRA_bboxs/BARRA_bboxs_actnsw"
year=2017
OUTDIR="/scratch/xe2/cb8590/barra_trees_s4_${year}_actnsw"

# Loop through each gpkg and submit a PBS job
for gpkg in "$GPKG_DIR"/*.gpkg; do
    fname=$(basename "$gpkg" .gpkg)
    echo "Submitting job for year $year, gpkg $gpkg, outdir $OUTDIR"
    qsub -N "predictions" \
         -v GPKG="$gpkg",OUTDIR="$OUTDIR",year="$year" \
         predictions.pbs
done