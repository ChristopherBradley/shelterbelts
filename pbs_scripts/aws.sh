#!/bin/bash

# I ran this command to get the subfolders: aws s3 ls s3://elvis-shelterbelts/nsw-elvis/elevation/lidar/z54/ | awk '{print $2}' | tr '\n' ' '

# Folder names
BASEURL="s3://elvis-shelterbelts/nsw-elvis/elevation/lidar/z54"
DEST_BASE="/scratch/xe2/cb8590/elvis-shelterbelts2/z54"
DONE=(Hatfield201411/ HawkerGate201708/ Kayrunnera201611/) # Used 'break' in the loop to test first before doing the rest of the loop
SUBFOLDERS=(LakeTandou201504/ LakeTandou201907/ LakeTandou202002/ LakeVictoria201503/ LakeVictoria202012/ Lindsay201302/ Lindsay202012/ Manara201510/ Manfred201411/ MenaMurtee201611/ Menindee201503/ Menindee201907/ MiddleCamp201503/ MiddleCamp202002/ Mildura201303/ Mildura202012/ Mildura202102/ Milparinka201707/ Monolon201608/ MountArrowsmith201708/ Mulurulu201411/ Murtee201705/ Nartooka201503/ Nartooka201907/ Nowingi201303/ Nowingi202102/ Nuchea201611/ Nyah202011/ OliveDowns201707/ Oxley202105/ Paika201303/ Paika202105/ Para201410/ Para201907/ Para202012/ Pooncarie201411/ Pooncarie201907/ Popiltah201503/ Popiltah202002/ Redan201503/ Redan202202/ Robinvale201303/ Robinvale202011/ Robinvale202102/ Scotia201503/ Smithville201708/ SwanHill201703/ SwanHill202011/ Taltingan201211/ Taltingan201612/ Taltingan202202/ Teilta201611/ Teryawynia201503/ Thackaringa201211/ Thackaringa201503/ Thackaringa202202/ ThurlooDowns201707/ Tibooburra201707/ Tongo201610/ Tongowoko201707/ Topar201611/ Turlee201411/ Urella201707/ Urisino201707/ Weimby201209/ Weimby201303/ Weimby202011/ Wentworth201302/ Wentworth201907/ Wentworth202002/ Wentworth202012/ WhiteCliffs201611/ Wilcannia201503/ WildDog201303/ WildDog202102/ Wonnaminta201611/ Yancannia201608/ Yantabangee201705/ Yantara201708/)

# # Single job at a time
# for f in "${SUBFOLDERS[@]}"; do
#     DEST="$DEST_BASE/$f"
#     FOLDER="$BASEURL/$f"
#     echo "Submitting job for $f"
#     qsub -v DEST="$DEST",folder="$FOLDER" aws.pbs
#     # break
# done

CHUNK_SIZE=2

for ((i=0; i<${#SUBFOLDERS[@]}; i+=CHUNK_SIZE)); do
    F1=${SUBFOLDERS[i]}
    F2=${SUBFOLDERS[i+1]}  # May be empty for the last job if odd number

    DEST1="$DEST_BASE/$F1"
    DEST2="$DEST_BASE/$F2"

    echo "Submitting job for $F1 $F2"
    qsub -N "aws_${F1%/}" -v DEST1="$DEST1",FOLDER1="$BASEURL/$F1",DEST2="$DEST2",FOLDER2="$BASEURL/$F2" aws.pbs

    # break
done