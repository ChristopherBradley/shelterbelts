#!/bin/bash

folder="/scratch/xe2/cb8590/Nick_Aus_treecover_10m"

for subfolder in "$folder"/subfolder_*; do
    echo "Submitting job for $subfolder"
    qsub -v subfolder="$subfolder" alphaearth.pbs
done

