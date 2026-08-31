#!/bin/bash
# Phase-2/3 driver: submit one gch_region.pbs per ~200 km region (148 total).
# Skips regions already merged (resume-safe). Set cleanup=0 for a sample region you want to keep
# all per-tile intermediates for; default cleanup=1 deletes inputs + per-tile intermediates after merge.
#   ./gch_region.sh            # all regions
#   SAMPLE=lat_34_lon_148 ./gch_region.sh   # keep intermediates for one sample region
PBSDIR="$(cd "$(dirname "$0")" && pwd)"
source /g/data/xe2/cb8590/miniconda/etc/profile.d/conda.sh
conda activate /g/data/xe2/cb8590/miniconda/envs/shelterbelts

BASE=/scratch/xe2/cb8590/gch_v2_ag_indices
SAMPLE=${SAMPLE:-lat_34_lon_148}

regions=$(python - <<'PY'
import geopandas as gpd, numpy as np, math
g=gpd.read_file('/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_ag.gpkg').set_crs(4326,allow_override=True)
c=g.geometry.centroid
regs=sorted(set(f'lat_{int(math.floor(abs(y)/2)*2)}_lon_{int(math.floor(x/2)*2)}' for y,x in zip(c.y.values,c.x.values)))
print('\n'.join(regs))
PY
)

cd "$PBSDIR"
n=0
for region in $regions; do
    if [ -f "${BASE}/indices/${region}_merged_shelter_categories.tif" ]; then
        echo "skip (merged exists): $region"; continue
    fi
    cleanup=1; [ "$region" = "$SAMPLE" ] && cleanup=0
    qsub -N "gchidx_${region}" -v region=${region},cleanup=${cleanup} gch_region.pbs
    n=$((n+1))
done
echo "Submitted $n region jobs (cleanup kept off for sample: $SAMPLE)"
