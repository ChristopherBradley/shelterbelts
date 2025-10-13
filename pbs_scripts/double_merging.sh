#!/bin/bash

# List of stubs (ACT folders)
# stubs=(DATA_727325 DATA_727319 DATA_727332 DATA_727334 DATA_727323 DATA_727327 DATA_727330 DATA_727317 DATA_727315 DATA_727321)
stubs=(DATA_717827 DATA_717837 DATA_717833 DATA_717831 DATA_717829 DATA_717843 DATA_717840 DATA_717850  \
             DATA_717847 DATA_717859 DATA_717863 DATA_717865 DATA_717856 DATA_717861 DATA_717867 DATA_717871)  # 4x4 test area


# Source and destination
# SRC_BASE="/scratch/xe2/cb8590/lidar"
# DEST="/scratch/xe2/cb8590/lidar_merged_ACT_2020_percentcover"
# # DEST="/scratch/xe2/cb8590/lidar_merged_ACT_2020_chm"

SRC_BASE="/scratch/xe2/cb8590/lidar_30km_old"
# DEST="/scratch/xe2/cb8590/lidar_merged_120km_percentcover"
# DEST="/scratch/xe2/cb8590/lidar_merged_120km_chm"
DEST="/scratch/xe2/cb8590/lidar_merged_120km_shelter/"

mkdir -p "$DEST"

for stub in "${stubs[@]}"; do
    # for f in "$SRC_BASE/$stub"/*height2m.tif; do
    # for f in "$SRC_BASE/$stub"/*res1.tif; do
    for f in "$SRC_BASE/$stub"/*linear_tifs_merged.tif; do
        [ -e "$f" ] || continue
        echo "Moving $f → $DEST"
        cp "$f" "$DEST/"
    done
done
