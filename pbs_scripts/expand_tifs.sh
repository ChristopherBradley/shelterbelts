#!/bin/bash

# folder_year=barra_trees_s4_2024
# stubs=(lat_34_lon_144 lat_34_lon_142 lat_28_lon_148 lat_28_lon_142 lat_32_lon_142 lat_28_lon_150 lat_30_lon_148 
#     lat_30_lon_140 lat_36_lon_148 lat_28_lon_144 lat_32_lon_144 lat_32_lon_146 lat_32_lon_150 lat_36_lon_150 
#     lat_28_lon_146 lat_32_lon_148 lat_28_lon_152 lat_36_lon_144 lat_34_lon_150 lat_34_lon_146 lat_30_lon_142 
#     lat_30_lon_150 lat_30_lon_152 lat_32_lon_140 lat_30_lon_144 lat_28_lon_140 lat_36_lon_146 lat_34_lon_148 
#     lat_30_lon_146 lat_32_lon_152 lat_34_lon_140)  # 2024

# stubs=(lat_34_lon_144 lat_34_lon_142)

year=2017
folder_year=barra_trees_s4_${year}_actnsw_4326
stubs=(lat_34_lon_144 lat_34_lon_142 lat_28_lon_148 lat_28_lon_142 lat_32_lon_142 lat_28_lon_150 lat_30_lon_148 
lat_30_lon_140 lat_36_lon_148 lat_28_lon_144 lat_32_lon_144 lat_32_lon_146 lat_32_lon_150 
lat_36_lon_150 lat_28_lon_146 lat_32_lon_148 lat_28_lon_152 lat_36_lon_144 lat_34_lon_150 
lat_34_lon_146 lat_30_lon_142 lat_30_lon_150 lat_30_lon_152 lat_32_lon_140 lat_30_lon_144 
lat_28_lon_140 lat_36_lon_146 lat_34_lon_148 lat_30_lon_146 lat_32_lon_152 lat_34_lon_140) # 2018 (they really should all be the same though ay?)

for stub in "${stubs[@]}"; do
    qsub -v stub="$stub",folder_year="$folder_year", expand_tifs.pbs
done