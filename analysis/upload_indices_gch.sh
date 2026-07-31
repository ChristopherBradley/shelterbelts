#!/bin/bash
# Upload the 3 GCH-v2 height-aware ag indices ImageCollections to ee-christopher-bradley,
# matching the 15 previously-uploaded collections (see upload_indices_15.sh).
# Method: GCH v2 canopy-height percent-cover input + height-aware WINDWARD shelter (+ opportunities).
# Datatypes: shelter_categories (MODE), shelter_distances (MEAN), opportunities (MODE).
# Pass "dry" as the first arg to preview without uploading.
set -uo pipefail

DRY=""
[ "${1:-}" = "dry" ] && DRY="--dry-run"

PY=/g/data/xe2/cb8590/miniconda/envs/shelterbelts/bin/python
SCRIPT=/home/147/cb8590/Projects/shelterbelts/analysis/upload_to_gee.py
SRC=/scratch/xe2/cb8590/gch_v2_ag_indices/indices
BUCKET=cb8590-shelterbelts-gee
PROJECT=ee-christopher-bradley
KEY=/home/147/cb8590/gee-uploader-key.json
METHOD=gch_windmethod

n=0
for pair in shelter_categories:MODE shelter_distances:MEAN opportunities:MODE; do
  d=${pair%%:*}; pyr=${pair##*:}
  n=$((n+1))
  suffix=_merged_${d}.tif
  coll=projects/${PROJECT}/assets/Aus2025_ag_${METHOD}_${d}
  echo "=================================================================="
  echo ">>> [$n/3] $(date '+%F %T')  ${d}  (pyramiding=${pyr})"
  echo ">>> collection=${coll}"
  "$PY" "$SCRIPT" "$SRC" \
    --suffix "$suffix" \
    --bucket "$BUCKET" \
    --gcs-prefix "indices_${METHOD}_${d}" \
    --collection "$coll" \
    --project "$PROJECT" \
    --key "$KEY" \
    --pyramiding "$pyr" \
    $DRY
done
echo "=================================================================="
echo ">>> ALL 3 GCH UPLOADS COMPLETE $(date '+%F %T')"
