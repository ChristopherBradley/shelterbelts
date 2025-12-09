#!/bin/bash


# NSW_folders=(DATA_717827 DATA_717837 DATA_717833 DATA_717831 DATA_717829 DATA_717843 DATA_717840 DATA_717850  \
#              DATA_717847 DATA_717859 DATA_717863 DATA_717865 DATA_717856 DATA_717861 DATA_717867 DATA_717871)  # 4x4 test area

# for f in "${NSW_folders[@]}"; do
#     folder="/scratch/xe2/cb8590/lidar_30km_old/$f/uint8_percentcover_res10_height2m"
#     outdir="/scratch/xe2/cb8590/lidar_30km_old/$f/linear_tifs"
#     qsub -v folder="$folder",outdir="$outdir" indices.pbs
# done

# NSW stubs
# param_stubs=(lat_28_lon_140 lat_28_lon_142 lat_28_lon_144 lat_28_lon_146 lat_28_lon_148 lat_28_lon_150 lat_28_lon_152 
#             lat_30_lon_140 lat_30_lon_142 lat_30_lon_144 lat_30_lon_146 lat_30_lon_148 lat_30_lon_150 lat_30_lon_152 
#             lat_32_lon_140 lat_32_lon_142 lat_32_lon_144 lat_32_lon_146 lat_32_lon_148 lat_32_lon_150 lat_32_lon_152 
#             lat_34_lon_140 lat_34_lon_142 lat_34_lon_144 lat_34_lon_146 lat_34_lon_148 lat_34_lon_150 lat_36_lon_144 
#             lat_36_lon_146 lat_36_lon_148 lat_36_lon_150)

# # Aus stubs
# param_stubs=(lat_24_lon_128 lat_14_lon_132 lat_34_lon_144 lat_24_lon_114 lat_10_lon_142 lat_34_lon_120 lat_24_lon_140 lat_34_lon_142 lat_26_lon_140 lat_26_lon_138 lat_36_lon_136 lat_14_lon_122 lat_16_lon_136 lat_20_lon_132 lat_32_lon_114 lat_16_lon_142 lat_26_lon_132 lat_22_lon_136 lat_24_lon_124 lat_28_lon_148 lat_10_lon_130 lat_22_lon_150 lat_34_lon_134 lat_28_lon_114 lat_22_lon_112 lat_28_lon_142 lat_32_lon_142 lat_20_lon_118 lat_28_lon_150 lat_18_lon_122 lat_16_lon_144 lat_28_lon_118 lat_12_lon_134 lat_32_lon_126 lat_42_lon_144 lat_8_lon_144 lat_24_lon_152 lat_24_lon_116 lat_30_lon_136 lat_30_lon_120 lat_38_lon_148 lat_20_lon_124 lat_14_lon_140 lat_30_lon_132 lat_32_lon_118 lat_28_lon_134 lat_18_lon_148 lat_14_lon_144 lat_32_lon_120 lat_22_lon_122 lat_34_lon_138 lat_14_lon_136 lat_30_lon_148 lat_22_lon_142 lat_30_lon_122 lat_38_lon_142 lat_30_lon_118 lat_16_lon_140 lat_18_lon_118 lat_30_lon_140 lat_20_lon_146 lat_26_lon_126 lat_20_lon_144 lat_26_lon_112 lat_42_lon_146 lat_16_lon_132 lat_14_lon_142 lat_36_lon_142 lat_36_lon_148 lat_26_lon_136 lat_28_lon_126 lat_24_lon_134 lat_16_lon_126 lat_30_lon_114 lat_22_lon_114 lat_18_lon_136 lat_22_lon_138 lat_30_lon_124 lat_12_lon_140 lat_32_lon_138 lat_10_lon_144 lat_28_lon_112 lat_28_lon_144 lat_32_lon_144 lat_28_lon_136 lat_16_lon_138 lat_20_lon_134 lat_24_lon_146 lat_12_lon_128 lat_32_lon_136 lat_24_lon_148 lat_20_lon_122 lat_18_lon_132 lat_26_lon_148 lat_32_lon_146 lat_28_lon_116 lat_20_lon_142 lat_34_lon_116 lat_20_lon_130 lat_26_lon_150 lat_32_lon_116 lat_32_lon_150 lat_36_lon_150 lat_22_lon_146 lat_32_lon_124 lat_18_lon_120 lat_16_lon_130 lat_16_lon_134 lat_38_lon_140 lat_28_lon_146 lat_22_lon_120 lat_8_lon_142 lat_32_lon_148 lat_18_lon_144 lat_30_lon_116 lat_22_lon_132 lat_10_lon_136 lat_28_lon_152 lat_22_lon_118 lat_28_lon_120 lat_18_lon_134 lat_26_lon_142 lat_38_lon_144 lat_36_lon_144 lat_24_lon_118 lat_42_lon_148 lat_34_lon_150 lat_34_lon_146 lat_26_lon_128 lat_30_lon_142 lat_30_lon_150 lat_30_lon_130 lat_26_lon_152 lat_20_lon_138 lat_12_lon_130 lat_36_lon_140 lat_20_lon_114 lat_26_lon_134 lat_28_lon_122 lat_20_lon_140 lat_24_lon_112 lat_20_lon_112 lat_22_lon_124 lat_22_lon_116 lat_16_lon_122 lat_20_lon_136 lat_12_lon_142 lat_26_lon_124 lat_32_lon_132 lat_28_lon_124 lat_28_lon_130 lat_36_lon_138 lat_40_lon_146 lat_16_lon_128 lat_30_lon_152 lat_14_lon_128 lat_32_lon_134 lat_32_lon_140 lat_40_lon_148 lat_16_lon_146 lat_22_lon_152 lat_24_lon_144 lat_18_lon_124 lat_24_lon_150 lat_26_lon_122 lat_24_lon_122 lat_26_lon_146 lat_16_lon_124 lat_30_lon_144 lat_20_lon_150 lat_18_lon_146 lat_32_lon_122 lat_30_lon_126 lat_30_lon_138 lat_34_lon_118 lat_12_lon_124 lat_34_lon_114 lat_34_lon_136 lat_30_lon_134 lat_32_lon_128 lat_18_lon_128 lat_8_lon_140 lat_26_lon_116 lat_26_lon_118 lat_14_lon_126 lat_20_lon_126 lat_28_lon_132 lat_18_lon_142 lat_10_lon_132 lat_18_lon_140 lat_26_lon_114 lat_12_lon_132 lat_28_lon_140 lat_36_lon_146 lat_10_lon_134 lat_24_lon_132 lat_20_lon_128 lat_38_lon_146 lat_28_lon_128 lat_18_lon_130 lat_24_lon_120 lat_22_lon_148 lat_40_lon_142 lat_22_lon_144 lat_22_lon_126 lat_26_lon_144 lat_40_lon_144 lat_24_lon_138 lat_30_lon_128 lat_28_lon_138 lat_34_lon_122 lat_20_lon_116 lat_20_lon_148 lat_24_lon_142 lat_18_lon_126 lat_26_lon_130 lat_26_lon_120 lat_22_lon_130 lat_24_lon_126 lat_34_lon_148 lat_12_lon_136 lat_24_lon_130 lat_22_lon_128 lat_20_lon_120 lat_22_lon_140 lat_14_lon_130 lat_18_lon_138 lat_30_lon_146 lat_14_lon_134 lat_24_lon_136 lat_10_lon_140 lat_14_lon_124 lat_32_lon_152 lat_34_lon_140 lat_22_lon_134 lat_12_lon_126) 
# # param_stubs=(lat_30_lon_152)  # Failed ones

# for param_stub in "${param_stubs[@]}"; do
#     qsub -v param_stub="$param_stub" indices.pbs
# done

# Trying lots of different parameters
qsub -v param_to_vary=max_shelterbelt_width,param_value=2 indices.pbs
qsub -v param_to_vary=max_shelterbelt_width,param_value=10 indices.pbs

qsub -v param_to_vary=min_shelterbelt_length,param_value=10 indices.pbs
qsub -v param_to_vary=min_shelterbelt_length,param_value=50 indices.pbs

qsub -v param_to_vary=min_patch_size,param_value=10 indices.pbs
qsub -v param_to_vary=min_patch_size,param_value=30 indices.pbs

qsub -v param_to_vary=min_core_size,param_value=20 indices.pbs
qsub -v param_to_vary=min_core_size,param_value=1000 indices.pbs

qsub -v param_to_vary=edge_size,param_value=1 indices.pbs
qsub -v param_to_vary=edge_size,param_value=10 indices.pbs

qsub -v param_to_vary=max_gap_size,param_value=0 indices.pbs
qsub -v param_to_vary=max_gap_size,param_value=3 indices.pbs

qsub -v param_to_vary=buffer_width,param_value=1 indices.pbs
qsub -v param_to_vary=buffer_width,param_value=5 indices.pbs

qsub -v param_to_vary=distance_threshold,param_value=5 indices.pbs
qsub -v param_to_vary=distance_threshold,param_value=20 indices.pbs

qsub -v param_to_vary=density_threshold,param_value=3 indices.pbs
qsub -v param_to_vary=density_threshold,param_value=10 indices.pbs

######

qsub -v param_to_vary=wind_method,param_value=WINDWARD indices.pbs
qsub -v param_to_vary=wind_method,param_value=MOST_COMMON indices.pbs
qsub -v param_to_vary=wind_method,param_value=ANY indices.pbs
qsub -v param_to_vary=wind_method,param_value=HAPPENED indices.pbs