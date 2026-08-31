#!/bin/bash
# Upload the GCH-v2 "final app" ag indices ImageCollections to ee-christopher-bradley.
# Companion to upload_indices_gch.sh (the old height-aware WINDWARD run) and upload_indices_15.sh
# (the predicted-tree run).
# 4 methods, but NOT 4x3 datatypes: only the datatype(s) each method actually merged exist on disk
# (gch_region_methods.pbs merges a different subset per method), so this loop uploads whatever it
# finds — 6 collections total for the default 4 methods:
#   less_windmethod       -> shelter_categories                    (MOST_COMMON, edge 3, mc 500,  height-aware)
#   default_windmethod    -> shelter_distances, opportunities      (WINDWARD,    edge 4, mc 1000, height-aware)
#   more_windmethod       -> shelter_categories                    (WINDWARD,    edge 5, mc 2000, no height)
#   default_percentmethod -> shelter_categories                    (density,     edge 4, mc 1000, no height)
# Collection name embeds the method + a _v2 tag, so these never collide with any older upload under
# the same method name (e.g. the pre-height-aware less_windmethod uploaded previously).
# Datatypes: shelter_categories (MODE), shelter_distances|densities (MEAN), opportunities (MODE).
# Uploads the MASKED rasters (indices_<method>_masked, ag-4km-tile mask only -> 255 nodata; no
# NLUM baked in, see gch_value_counts_methods.pbs), which need gch_value_counts_methods.pbs to have
# run. Set RAW=1 to upload the unmasked indices_<method> instead, matching what upload_indices_gch.sh
# did for the height-aware run.
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
VERSION_TAG=${VERSION_TAG:-v3}

methods=${METHODS:-"less_windmethod default_windmethod more_windmethod default_percentmethod"}
# Masked rasters use 255 outside the ag tiles; declare it so GEE renders that area transparent.
# (RAW=1 uploads are unmasked and have no such fill, so no missingData is declared for them.)
if [ "${RAW:-0}" = "1" ]; then NODATA_ARG=""; else NODATA_ARG="--nodata 255"; fi

# First pass: only count (method, datatype) pairs that actually have a merged raster, so the
# [n/total] progress line is accurate regardless of which subset each method produced.
total=0
for m in $methods; do
  case $m in
    *windmethod)    dist=shelter_distances;;
    *percentmethod) dist=shelter_densities;;
  esac
  if [ "${RAW:-0}" = "1" ]; then SRC=${BASE}/indices_${m}; else SRC=${BASE}/indices_${m}_masked; fi
  for d in shelter_categories $dist opportunities; do
    ls ${SRC}/*_merged_${d}.tif >/dev/null 2>&1 && total=$((total+1))
  done
done

n=0
for m in $methods; do
  case $m in
    *windmethod)    dist=shelter_distances;;
    *percentmethod) dist=shelter_densities;;
  esac
  if [ "${RAW:-0}" = "1" ]; then SRC=${BASE}/indices_${m}; else SRC=${BASE}/indices_${m}_masked; fi
  for d in shelter_categories $dist opportunities; do
    suffix=_merged_${d}.tif
    ls ${SRC}/*${suffix} >/dev/null 2>&1 || { echo "skip (no ${suffix} in $SRC)"; continue; }
    pyr=MEAN; [ "$d" = "shelter_categories" ] && pyr=MODE; [ "$d" = "opportunities" ] && pyr=MODE
    n=$((n+1))
    coll=projects/${PROJECT}/assets/Aus2025_ag_gch_${m}_${VERSION_TAG}_${d}
    echo "=================================================================="
    echo ">>> [$n/$total] $(date '+%F %T')  ${m} / ${d}  (pyramiding=${pyr})"
    echo ">>> src=${SRC}"
    echo ">>> collection=${coll}"
    "$PY" "$SCRIPT" "$SRC" \
      --suffix "$suffix" \
      --bucket "$BUCKET" \
      --gcs-prefix "indices_gch_${m}_${VERSION_TAG}_${d}" \
      --collection "$coll" \
      --project "$PROJECT" \
      --key "$KEY" \
      --pyramiding "$pyr" \
      ${NODATA_ARG} \
      $DRY
  done
done
echo "=================================================================="
echo ">>> ALL $n GCH METHOD UPLOADS COMPLETE $(date '+%F %T')"
