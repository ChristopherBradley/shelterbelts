#!/bin/bash

# Directory containing the split gpkg files
year=2020
state=aus
# GPKG_DIR="/g/data/xe2/cb8590/Outlines/BARRA_bboxs/BARRA_bboxs_${state}_4326_weightings_median_${year}_attempt4"
# GPKG_DIR="/g/data/xe2/cb8590/Outlines/BARRA_bboxs/BARRA_bboxs_${state}_noxy_df_4326_${year}"
GPKG_DIR="/g/data/xe2/cb8590/Outlines/BARRA_bboxs/BARRA_bboxs_${state}_noxy_df_4326_${year}_attempt4"
OUTDIR="/scratch/xe2/cb8590/barra_trees_s4_${state}_noxy_df_4326_${year}"

# Loop through each gpkg and submit a PBS job
for gpkg in "$GPKG_DIR"/*.gpkg; do
    echo "Submitting job for year $year, gpkg $gpkg, outdir $OUTDIR"
    qsub -N "predictions" \
         -v GPKG="$gpkg",OUTDIR="$OUTDIR",year="$year" \
         predictions.pbs
done