#!/bin/bash

FOLDER=/scratch/xe2/cb8590/Nick_sentinel/tiles_todo

# Move all files back up one level
find "$FOLDER" -type f -path "$FOLDER/subfolder_*/*" -exec mv {} "$FOLDER" \;

# Remove the now-empty subfolders
find "$FOLDER" -type d -name "subfolder_*" -exec rmdir {} +
