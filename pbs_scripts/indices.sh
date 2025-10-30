#!/bin/bash


# NSW_folders=(DATA_717827 DATA_717837 DATA_717833 DATA_717831 DATA_717829 DATA_717843 DATA_717840 DATA_717850  \
#              DATA_717847 DATA_717859 DATA_717863 DATA_717865 DATA_717856 DATA_717861 DATA_717867 DATA_717871)  # 4x4 test area

# for f in "${NSW_folders[@]}"; do
#     folder="/scratch/xe2/cb8590/lidar_30km_old/$f/uint8_percentcover_res10_height2m"
#     outdir="/scratch/xe2/cb8590/lidar_30km_old/$f/linear_tifs"
#     qsub -v folder="$folder",outdir="$outdir" indices.pbs
# done

param_stubs=(lat_28_lon_140 lat_28_lon_142 lat_28_lon_144 lat_28_lon_146 lat_28_lon_148 lat_28_lon_150 lat_28_lon_152 
            lat_30_lon_140 lat_30_lon_142 lat_30_lon_144 lat_30_lon_146 lat_30_lon_148 lat_30_lon_150 lat_30_lon_152 
            lat_32_lon_140 lat_32_lon_142 lat_32_lon_144 lat_32_lon_146 lat_32_lon_148 lat_32_lon_150 lat_32_lon_152 
            lat_34_lon_140 lat_34_lon_142 lat_34_lon_144 lat_34_lon_146 lat_34_lon_148 lat_34_lon_150 lat_36_lon_144 
            lat_36_lon_146 lat_36_lon_148 lat_36_lon_150)

# param_stubs=(lat_30_lon_152)  # Failed ones

for param_stub in "${param_stubs[@]}"; do
    qsub -v param_stub="$param_stub" indices.pbs
done