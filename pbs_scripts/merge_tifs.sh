#!/bin/bash
# Merge driver for the 2025 ag indices full-run: one merge_tifs.pbs job per per-tile folder,
# which merges all present shelter/opportunity tif types + combines the patch_metrics CSVs.
# Edit GLOB for the phase: *windmethod (phase 1) or *percentmethod (phase 2).

BASE=/scratch/xe2/cb8590/barra_trees_s4_ag_noxy_df_4326_2025

# PHASE 1 (wind) — done:
# GLOB="lat_*_*windmethod"
# JOBLOG=/scratch/xe2/cb8590/fullrun_phase1_merge_jobids.txt
# PHASE 2 (density):
GLOB="lat_*_*percentmethod"
JOBLOG=/scratch/xe2/cb8590/fullrun_phase2_merge_jobids.txt

# Skip a folder whose merged outputs already exist in the parent (resume-safe / skips the test folder).
> "$JOBLOG"
n=0
for d in $(ls -d ${BASE}/indices/${GLOB} 2>/dev/null); do
    name=$(basename "$d")
    if [ -f "${BASE}/indices/${name}_merged_shelter_categories.tif" ] && [ -f "${BASE}/indices/${name}_patch_metrics.csv" ]; then
        continue  # already merged
    fi
    jid=$(qsub -v base_dir="$d" merge_tifs.pbs)
    echo "$jid $name" >> "$JOBLOG"
    n=$((n+1))
done
echo "Submitted $n merge jobs (logged to $JOBLOG)"
