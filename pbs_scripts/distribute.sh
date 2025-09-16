#!/bin/bash

# Usage: ./distribute.sh <folder> <n_subfolders>

FOLDER="$1"
N="$2"

# Get list of files (excluding directories)
FILES=("$FOLDER"/*)
TOTAL=${#FILES[@]}

# Create subfolders
for i in $(seq 1 "$N"); do
  mkdir -p "$FOLDER/subfolder_$i"
done

# Distribute files
i=0
for file in "${FILES[@]}"; do
  folder_num=$(( (i % N) + 1 ))
  mv "$file" "$FOLDER/subfolder_$folder_num/"
  ((i++))
done