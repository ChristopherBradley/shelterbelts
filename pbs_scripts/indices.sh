#!/bin/bash


NSW_folders=(DATA_717827 DATA_717837 DATA_717833 DATA_717831 DATA_717829 DATA_717843 DATA_717840 DATA_717850  \
             DATA_717847 DATA_717859 DATA_717863 DATA_717865 DATA_717856 DATA_717861 DATA_717867 DATA_717871)  # 4x4 test area

for f in "${NSW_folders[@]}"; do
    folder="/scratch/xe2/cb8590/lidar_30km_old/$f/uint8_percentcover_res10_height2m"
    outdir="/scratch/xe2/cb8590/lidar_30km_old/$f/linear_tifs"
    qsub -N "indices_${f}" -v folder="$folder",outdir="$outdir" indices.pbs
done