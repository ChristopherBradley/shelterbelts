#!/bin/bash

# Usage: ./distribute.sh <folder> <max_files_per_subfolder>

FOLDER="$1"
MAX="$2"

# Skip if distribution already exists
if [ -d "$FOLDER/subfolder_1" ]; then
    echo "Distribution already exists in $FOLDER, skipping..."
    exit 0
fi

# Get list of files (excluding directories)
# FILES=("$FOLDER"/*)
FILES=("$FOLDER"/*.tiff)  # Just the pickle files
TOTAL=${#FILES[@]}

# Distribute files into subfolders with up to MAX files each
i=0
folder_idx=1
mkdir -p "$FOLDER/subfolder_$folder_idx"

for file in "${FILES[@]}"; do
  mv "$file" "$FOLDER/subfolder_$folder_idx/"
  ((i++))

  if (( i >= MAX )); then
    ((folder_idx++))
    mkdir -p "$FOLDER/subfolder_$folder_idx"
    i=0
  fi
done