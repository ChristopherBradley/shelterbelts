#!/bin/bash
# Full-run driver: submit indices_methods.pbs for a set of methods across all 148 ag regions.
# The 2025 ag indices run is split into 2 phases to stay under the scratch inode ceiling
# (only 3 methods' tiles on disk at a time): PHASE 1 = wind methods, PHASE 2 = density methods.
# After each phase's folders are merged (merge_tifs.sh) + patch_metrics combined, delete the
# per-tile intermediates before starting the next phase.
#
# Edit METHODS + JOBLOG for the phase you want, then run ./indices.sh
# Skips any (region,method) whose output folder already exists (idempotent / resume-safe).

BASE=/scratch/xe2/cb8590/barra_trees_s4_ag_noxy_df_4326_2025

# PHASE 1 (wind) — done:
# METHODS="default_windmethod more_windmethod less_windmethod"
# JOBLOG=/scratch/xe2/cb8590/fullrun_phase1_jobids.txt
# PHASE 2 (density):
METHODS="default_percentmethod more_percentmethod less_percentmethod"
JOBLOG=/scratch/xe2/cb8590/fullrun_phase2_jobids.txt

> "$JOBLOG"
regions=$(ls -d ${BASE}/expanded/lat_*/ | xargs -n1 basename)
n=0
for method in $METHODS; do
    for region in $regions; do
        outdir=${BASE}/indices/${region}_${method}
        if [ -d "$outdir" ]; then
            echo "skip (exists): ${region} ${method}"
            continue
        fi
        jid=$(qsub -N "idx_${method}_${region}" \
            -v folder=${BASE}/expanded/${region},outdir=${outdir},param_stub=${region},method=${method} \
            indices_methods.pbs)
        echo "$jid ${region} ${method}" >> "$JOBLOG"
        n=$((n+1))
    done
done
echo "Submitted $n jobs (logged to $JOBLOG)"

# For historical single-method reference, see git history of this file (old param_stubs array + indices.pbs).
