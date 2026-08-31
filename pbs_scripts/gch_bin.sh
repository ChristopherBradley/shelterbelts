#!/bin/bash
# Phase-1 driver: precompute the needed-tile list, then submit binning jobs over slices of it.
# Each job bins ~CHUNK tiles (~45 s each). Idempotent — safe to re-run to fill gaps.
#   ./gch_bin.sh
PBSDIR="$(cd "$(dirname "$0")" && pwd)"
source /g/data/xe2/cb8590/miniconda/etc/profile.d/conda.sh
conda activate /g/data/xe2/cb8590/miniconda/envs/shelterbelts

LIST=/scratch/xe2/cb8590/gch_v2_ag_indices/needed_gch_tiles.txt
[ -f "$LIST" ] || (cd "$PBSDIR/../analysis" && python gch_bin.py --make_list)
N=$(wc -l < "$LIST")
CHUNK=${CHUNK:-150}      # ~150 tiles/job * ~45s ~= <2h
echo "Binning $N needed GCH tiles in chunks of $CHUNK"
cd "$PBSDIR"
jids=""
for start in $(seq 0 $CHUNK $((N-1))); do
    end=$((start+CHUNK))
    jid=$(qsub -N "gchbin_${start}" -v start=${start},end=${end} gch_bin.pbs)
    jids="$jids $jid"
done
echo "Submitted binning jobs:$jids"
