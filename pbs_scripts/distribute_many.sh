#!/bin/bash

# Usage: ./distribute_many.sh <max_files_per_subfolder>


# List of NSW folders
NSW_folders=(DATA_717827 DATA_717837 DATA_717833 DATA_717831 DATA_717829 DATA_717843 DATA_717840 DATA_717850 DATA_717847 \
             DATA_717859 DATA_717863 DATA_717865 DATA_717856 DATA_717861 DATA_717867 DATA_717871)

BASE="/scratch/xe2/cb8590/lidar"

for folder in "${NSW_folders[@]}"; do    
    LAZ_FOLDER="$BASE/$folder/laz_files/NSW Government - Spatial Services/Point Clouds/AHD"

    echo "Distributing files in $LAZ_FOLDER"
    ./distribute.sh "$LAZ_FOLDER" 100
done