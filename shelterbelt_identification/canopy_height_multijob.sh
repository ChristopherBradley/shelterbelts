#!/bin/bash

# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/shelterbelts 

# Automatically submit a whole bunch of jobs for lots of locations
batch_filename="/g/data/xe2/cb8590/Nick_outlines/canopy_height_batches.csv"

# Loop through each line in the file
while IFS=, read -r csv; do
    # Trim any leading or trailing whitespace
    csv=$(echo $csv | xargs)

    ## Run first job
    job_id1=$(qsub -v wd=$wd,csv=$csv canopy_height_singlejob.pbs)
    echo "job submitted with ID $job_id1"

done < "$batch_filename"