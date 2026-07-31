#!/bin/bash
# Upload the 15 indices ImageCollections (5 methods x 3 datatypes) to ee-christopher-bradley.
# Methods: less/more windmethod + default/less/more percentmethod (default_windmethod already uploaded).
# Datatypes: shelter_categories (MODE), shelter_distances|densities (MEAN), opportunities (MODE).
# Pass "dry" as the first arg to preview without uploading.
set -uo pipefail

DRY=""
[ "${1:-}" = "dry" ] && DRY="--dry-run"

PY=/g/data/xe2/cb8590/miniconda/envs/shelterbelts/bin/python
SCRIPT=/home/147/cb8590/Projects/shelterbelts/analysis/upload_to_gee.py
SRC=/scratch/xe2/cb8590/barra_trees_s4_ag_noxy_df_4326_2025/indices
BUCKET=cb8590-shelterbelts-gee
PROJECT=ee-christopher-bradley
KEY=/home/147/cb8590/gee-uploader-key.json

methods="less_windmethod more_windmethod default_percentmethod less_percentmethod more_percentmethod"
n=0
for m in $methods; do
  case $m in
    *windmethod)    dist=shelter_distances;;
    *percentmethod) dist=shelter_densities;;
  esac
  for pair in shelter_categories:MODE ${dist}:MEAN opportunities:MODE; do
    d=${pair%%:*}; pyr=${pair##*:}
    n=$((n+1))
    suffix=_${m}_merged_${d}.tif
    coll=projects/${PROJECT}/assets/Aus2025_ag_${m}_${d}
    echo "=================================================================="
    echo ">>> [$n/15] $(date '+%F %T')  ${m} / ${d}  (pyramiding=${pyr})"
    echo ">>> collection=${coll}"
    "$PY" "$SCRIPT" "$SRC" \
      --suffix "$suffix" \
      --bucket "$BUCKET" \
      --gcs-prefix "indices_${m}_${d}" \
      --collection "$coll" \
      --project "$PROJECT" \
      --key "$KEY" \
      --pyramiding "$pyr" \
      $DRY
  done
done
echo "=================================================================="
echo ">>> ALL 15 UPLOADS COMPLETE $(date '+%F %T')"
