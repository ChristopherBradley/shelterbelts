#!/bin/bash

for f in /scratch/xe2/cb8590/barra_trees_s4_*_actnsw_4326/subfolders/*merged_predicted*.tif; do
  [ -e "$f" ] || continue

  year=$(echo "$f" | sed -E 's|.*/barra_trees_s4_([0-9]{4})_actnsw_4326/.*|\1|')    # extract year from the grandparent folder name
  dir=$(dirname "$f")
  base=$(basename "$f" .tif)
  new="${dir}/${base}_${year}.tif"

  mv -v -- "$f" "$new"
done
