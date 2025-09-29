#!/bin/bash

CHUNK_DIR=/scratch/xe2/cb8590/Nick_sentinel/chunks/

for f in $CHUNK_DIR/tiff_footprints_chunk_*.gpkg; do
    qsub -v filename="$f" sentinel.pbs
done