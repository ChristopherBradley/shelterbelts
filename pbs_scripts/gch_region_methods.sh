#!/bin/bash
# Driver for the GCH "final app" method rerun: submit gch_region_methods.pbs for every
# (region, method) pair — 148 regions x N methods.
# Skips any pair already merged (resume-safe). Each job cuts its own inputs, so methods
# are fully independent and can run concurrently.
#   ./gch_region_methods.sh                              # everything not yet done (default 4 methods)
#   REGIONS="lat_10_lon_142 lat_30_lon_150" ./gch_region_methods.sh   # pilot on named regions
#   METHODS=less_windmethod ./gch_region_methods.sh      # one method only
#   CLEANUP=0 ./gch_region_methods.sh                    # keep per-tile intermediates
PBSDIR="$(cd "$(dirname "$0")" && pwd)"
source /g/data/xe2/cb8590/miniconda/etc/profile.d/conda.sh
conda activate /g/data/xe2/cb8590/miniconda/envs/shelterbelts

BASE=/scratch/xe2/cb8590/gch_v2_ag_indices
METHODS=${METHODS:-"less_windmethod default_windmethod more_windmethod default_percentmethod"}
CLEANUP=${CLEANUP:-1}
JOBLOG=${JOBLOG:-${BASE}/gch_methods_jobids.txt}

# Each method's primary merged output (must match gch_region_methods.pbs's merge_suffixes[0]),
# used both as the resume-skip check and to build a short job-name tag.
declare -A PRIMARY_SUFFIX=(
    [less_windmethod]=shelter_categories
    [more_windmethod]=shelter_categories
    [default_percentmethod]=shelter_categories
    [default_windmethod]=shelter_categories
)
declare -A TAG=(
    [less_windmethod]=lw
    [default_windmethod]=dw
    [more_windmethod]=mw
    [default_percentmethod]=dp
)

regions=${REGIONS:-$(python - <<'PY'
import geopandas as gpd, math
g=gpd.read_file('/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_ag.gpkg').set_crs(4326,allow_override=True)
c=g.geometry.centroid
regs=sorted(set(f'lat_{int(math.floor(abs(y)/2)*2)}_lon_{int(math.floor(x/2)*2)}' for y,x in zip(c.y.values,c.x.values)))
print('\n'.join(regs))
PY
)}

# Job names of anything already queued/running, so a re-run never double-submits a region in flight.
inflight=$(qstat -f 2>/dev/null | awk '/Job_Name = /{print $3}')   # -f ignores -u on gadi; defaults to own jobs

cd "$PBSDIR"
n=0
for method in $METHODS; do
    primary=${PRIMARY_SUFFIX[$method]:-shelter_categories}
    tag=${TAG[$method]:-xx}
    for region in $regions; do
        if [ -f "${BASE}/indices_${method}/${region}_merged_${primary}.tif" ]; then
            echo "skip (merged exists): $region $method"; continue
        fi
        if grep -qx "gch_${tag}_${region}" <<<"$inflight"; then
            echo "skip (in queue): $region $method"; continue
        fi
        jid=$(qsub -N "gch_${tag}_${region}" \
            -v region=${region},method=${method},cleanup=${CLEANUP} gch_region_methods.pbs)
        echo "$jid ${region} ${method}" >> "$JOBLOG"
        n=$((n+1))
    done
done
echo "Submitted $n jobs (logged to $JOBLOG)"
