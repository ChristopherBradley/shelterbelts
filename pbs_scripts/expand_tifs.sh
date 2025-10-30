#!/bin/bash

stubs=(lat_34_lon_144 lat_34_lon_142 lat_28_lon_148 lat_28_lon_142 lat_32_lon_142 lat_28_lon_150 lat_30_lon_148 
    lat_30_lon_140 lat_36_lon_148 lat_28_lon_144 lat_32_lon_144 lat_32_lon_146 lat_32_lon_150 lat_36_lon_150 
    lat_28_lon_146 lat_32_lon_148 lat_28_lon_152 lat_36_lon_144 lat_34_lon_150 lat_34_lon_146 lat_30_lon_142 
    lat_30_lon_150 lat_30_lon_152 lat_32_lon_140 lat_30_lon_144 lat_28_lon_140 lat_36_lon_146 lat_34_lon_148 
    lat_30_lon_146 lat_32_lon_152 lat_34_lon_140)

# stubs=(lat_34_lon_144 lat_34_lon_142)

for stub in "${stubs[@]}"; do
    qsub -v stub="$stub" expand_tifs.pbs
done