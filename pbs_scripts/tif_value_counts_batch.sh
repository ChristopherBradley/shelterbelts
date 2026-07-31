#!/bin/bash
# Submit tif_value_counts.pbs for every wind-method / output-type combination
# over the 2025 ag indices, optionally split by a region outline.
#
# 18 combos = {default,less,more} x {wind,percent}method x 3 output types
#   windmethod   -> opportunities, shelter_categories, shelter_distances
#   percentmethod-> opportunities, shelter_categories, shelter_densities
#
# Usage:
#   ./tif_value_counts_batch.sh global                 # no zones (18 jobs)
#   ./tif_value_counts_batch.sh states                 # split by a region (18 jobs)
#   ./tif_value_counts_batch.sh states grdc nrm ibra lga
#   ./tif_value_counts_batch.sh all                    # global + all 5 regions
#
# Outputs: /scratch/xe2/cb8590/tif_value_counts/<stage>/<combo>.csv

set -euo pipefail

FOLDER=/g/data/xe2/cb8590/shelterbelt_outputs/barra_trees_s4_ag_noxy_df_4326_2025/indices
OUTROOT=/scratch/xe2/cb8590/tif_value_counts
PBS=/home/147/cb8590/Projects/shelterbelts2/pbs_scripts/tif_value_counts.pbs

# 18 suffixes (without the trailing .tif)
suffixes=()
for m in default_windmethod less_windmethod more_windmethod; do
  for t in opportunities shelter_categories shelter_distances; do
    suffixes+=("${m}_merged_${t}")
  done
done
for m in default_percentmethod less_percentmethod more_percentmethod; do
  for t in opportunities shelter_categories shelter_densities; do
    suffixes+=("${m}_merged_${t}")
  done
done

# region stage -> "outline_file:zone_column"
declare -A ZONE
ZONE[states]="/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp:STE_NAME21"
ZONE[grdc]="/g/data/xe2/cb8590/Outlines/grdc_regions.gpkg:AEZ"
ZONE[nrm]="/g/data/xe2/cb8590/Outlines/NRM_clusters.zip:label"
ZONE[ibra]="/g/data/xe2/cb8590/Outlines/ibra7_regions.gpkg:REG_NAME_7"
ZONE[lga]="/g/data/xe2/cb8590/Outlines/LGA_2025_AUST_GDA2020.zip:LGA_NAME25"

submit_stage() {
  local stage="$1"
  local outdir="$OUTROOT/$stage"
  mkdir -p "$outdir"
  local zfile="" zcol=""
  if [ "$stage" != "global" ]; then
    IFS=':' read -r zfile zcol <<< "${ZONE[$stage]}"
  fi
  local n=0
  for s in "${suffixes[@]}"; do
    local out="$outdir/${s}.csv"
    if [ "$stage" == "global" ]; then
      qsub -N "vc_${stage}" \
        -v tif_folder="$FOLDER",suffix="${s}.tif",output_csv="$out" "$PBS"
    else
      qsub -N "vc_${stage}" \
        -v tif_folder="$FOLDER",suffix="${s}.tif",output_csv="$out",zones_file="$zfile",zone_col="$zcol" "$PBS"
    fi
    n=$((n+1))
  done
  echo "Submitted $n jobs for stage '$stage' -> $outdir"
}

stages=("$@")
if [ "${#stages[@]}" -eq 0 ]; then
  echo "Usage: $0 <global|states|grdc|nrm|ibra|lga|all> ..."; exit 1
fi
for stage in "${stages[@]}"; do
  if [ "$stage" == "all" ]; then
    for st in global states grdc nrm ibra lga; do submit_stage "$st"; done
  else
    submit_stage "$stage"
  fi
done
