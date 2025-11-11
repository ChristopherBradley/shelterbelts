#!/bin/bash



param_stubs=(lat_28_lon_140 lat_28_lon_142 lat_28_lon_144 lat_28_lon_146 lat_28_lon_148 lat_28_lon_150 lat_28_lon_152 
            lat_30_lon_140 lat_30_lon_142 lat_30_lon_144 lat_30_lon_146 lat_30_lon_148 lat_30_lon_150 lat_30_lon_152 
            lat_32_lon_140 lat_32_lon_142 lat_32_lon_144 lat_32_lon_146 lat_32_lon_148 lat_32_lon_150 lat_32_lon_152 
            lat_34_lon_140 lat_34_lon_142 lat_34_lon_144 lat_34_lon_146 lat_34_lon_148 lat_34_lon_150 lat_36_lon_144 
            lat_36_lon_146 lat_36_lon_148 lat_36_lon_150)

for param_stub in "${param_stubs[@]}"; do
    qsub -v param_stub="$param_stub" opportunities.pbs
done