#!/bin/bash

# Submit one snakes_bbox.pbs job per ACT2020 laz tile listed by
# analysis/snakes_bbox_prep.py. All tiles run in parallel (4GB/1cpu each) so the
# ~254-tile ACT2020 run over snakes_bbox finishes in ~1 hour.

LISTFILE="/scratch/xe2/cb8590/lidar_processing/snakes/snakes_act_laz.txt"
OUTDIR="/scratch/xe2/cb8590/lidar_processing/snakes/chms"

mkdir -p "$OUTDIR"

n=0
while IFS= read -r laz; do
    [ -z "$laz" ] && continue
    stub=$(basename "$laz" .laz)
    qsub -v LAZ="$laz",STUB="$stub",OUTDIR="$OUTDIR" snakes_bbox.pbs
    n=$((n+1))
done < "$LISTFILE"
echo "Submitted $n jobs"
