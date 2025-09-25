#!/bin/bash



## Run merge_lidar.pbs on a single folder
# stub=DATA_587060, DATA_587065
# qsub -v stub="$stub" merge_lidar.pbs

## Run it on all the folders
# stubs=(DATA_587068 DATA_587081 DATA_587083 DATA_587085 \
#        DATA_587109 DATA_587122 DATA_587124 DATA_587133 \
#        DATA_587116 DATA_587120 DATA_587127 DATA_587060 \
#        DATA_587065 DATA_587130 DATA_587000 DATA_586204)
# stubs=(DATA_709828)
stubs=(DATA_717827 DATA_717837 DATA_717833 DATA_717831 DATA_717829 DATA_717843 DATA_717840 DATA_717850 DATA_717847 \
             DATA_717859 DATA_717863 DATA_717865 DATA_717856 DATA_717861 DATA_717867 DATA_717871)
for stub in "${stubs[@]}"; do
    qsub -v stub="$stub" merge_lidar.pbs
done