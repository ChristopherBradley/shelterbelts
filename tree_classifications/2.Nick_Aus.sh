#!/bin/bash

# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/shelterbelts/tree_classifications 

indir='/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
outdir='/scratch/xe2/cb8590/Nick_sentinel'

# Automatically submit a whole bunch of jobs for lots of locations
coordinates_file="/g/data/xe2/cb8590/Nick_outlines/gdf_x100.csv"
# coordinates_file="/g/data/xe2/cb8590/Nick_outlines/gdf_x5.csv"
# coordinates_file="/g/data/xe2/cb8590/Nick_outlines/gdf_01001.csv"

# Loop through each line in the file
first_line=true
while IFS=, read -r tif year; do
    # Skip the header of the csv
    if $first_line; then
        first_line=false
        continue
    fi

    # Trim any leading or trailing whitespace
    tif=$(echo $tif | xargs)
    year=$(echo $year | xargs)

    ## Run first job
    job_id1=$(qsub -v wd=$wd,indir=$indir,outdir=$outdir,tif=$tif,year=$year run_sentinel.sh)
    echo "First job submitted for stub $stub with ID $job_id1"

    # echo "filename: $tif, year $year"

done < "$coordinates_file"