#!/bin/bash
# Upload the GCH-v2 threshold-sensitivity ag indices ImageCollections to ee-christopher-bradley:
# N methods x 3 datatypes = 3N collections. Companion to upload_indices_gch.sh (the height-aware
# WINDWARD run) and upload_indices_15.sh (the predicted-tree run).
# Method: GCH v2 canopy-height percent-cover input, NOT height-aware (distances in metres).
#   less_percentmethod — edge_size 3, min_core_size 500,  density shelter          -> shelter_densities
#   more_windmethod    — edge_size 5, min_core_size 2000, WINDWARD,     wt 15      -> shelter_distances
#   less_windmethod    — edge_size 3, min_core_size 500,  MOST_COMMON,  wt 25      -> shelter_distances
# Collection name embeds the method, so a new method automatically gets its own new layer/collection.
# Datatypes: shelter_categories (MODE), shelter_distances|densities (MEAN), opportunities (MODE).
# Uploads the MASKED rasters (indices_<method>_masked, gaps between ag tiles = 255 nodata), which
# need gch_value_counts_methods.pbs to have run. Set RAW=1 to upload the unmasked indices_<method>
# instead, matching what upload_indices_gch.sh did for the height-aware run.
# Pass "dry" as the first arg to preview without uploading.
set -uo pipefail

DRY=""
[ "${1:-}" = "dry" ] && DRY="--dry-run"

PY=/g/data/xe2/cb8590/miniconda/envs/shelterbelts/bin/python
SCRIPT=/home/147/cb8590/Projects/shelterbelts/analysis/upload_to_gee.py
BASE=/scratch/xe2/cb8590/gch_v2_ag_indices
BUCKET=cb8590-shelterbelts-gee
PROJECT=ee-christopher-bradley
KEY=/home/147/cb8590/gee-uploader-key.json

methods=${METHODS:-"less_percentmethod more_windmethod"}
# Masked rasters use 255 outside the ag tiles; declare it so GEE renders that area transparent.
# (RAW=1 uploads are unmasked and have no such fill, so no missingData is declared for them.)
if [ "${RAW:-0}" = "1" ]; then NODATA_ARG=""; else NODATA_ARG="--nodata 255"; fi
n=0
for m in $methods; do
  case $m in
    *windmethod)    dist=shelter_distances;;
    *percentmethod) dist=shelter_densities;;
  esac
  if [ "${RAW:-0}" = "1" ]; then SRC=${BASE}/indices_${m}; else SRC=${BASE}/indices_${m}_masked; fi
  for pair in shelter_categories:MODE ${dist}:MEAN opportunities:MODE; do
    d=${pair%%:*}; pyr=${pair##*:}
    n=$((n+1))
    suffix=_merged_${d}.tif
    coll=projects/${PROJECT}/assets/Aus2025_ag_gch_${m}_${d}
    echo "=================================================================="
    echo ">>> [$n/6] $(date '+%F %T')  ${m} / ${d}  (pyramiding=${pyr})"
    echo ">>> src=${SRC}"
    echo ">>> collection=${coll}"
    "$PY" "$SCRIPT" "$SRC" \
      --suffix "$suffix" \
      --bucket "$BUCKET" \
      --gcs-prefix "indices_gch_${m}_${d}" \
      --collection "$coll" \
      --project "$PROJECT" \
      --key "$KEY" \
      --pyramiding "$pyr" \
      ${NODATA_ARG} \
      $DRY
  done
done
echo "=================================================================="
echo ">>> ALL 6 GCH METHOD UPLOADS COMPLETE $(date '+%F %T')"
