#!/bin/bash
# Submit a bunch of indices evaluation jobs.
# Claude-generated and rewritten due to bugs, might want to tidy up later.

set -e

STUBS_CSV=${STUBS_CSV:-/scratch/xe2/cb8590/Adjacent_eval_tiles/adjacent_stubs.csv}
GT_DIR=${GT_DIR:-/scratch/xe2/cb8590/Adjacent_eval_tiles}
OUT_BASE=${OUT_BASE:-/scratch/xe2/cb8590/adjacent_eval_indices/indices}
CHUNK_DIR=${CHUNK_DIR:-/scratch/xe2/cb8590/Adjacent_eval_tiles/chunks}
NCHUNKS=${NCHUNKS:-16}

mkdir -p "$CHUNK_DIR"
rm -f "$CHUNK_DIR"/chunk_*.csv
# Split the data rows (skip header) into NCHUNKS files, then give each its header.
# (Header all chunks BEFORE submitting so a transient qsub failure can't leave
# some chunks header-less.)
tail -n +2 "$STUBS_CSV" > "$CHUNK_DIR/_rows.txt"
split -n l/"$NCHUNKS" -d --additional-suffix=.csv "$CHUNK_DIR/_rows.txt" "$CHUNK_DIR/chunk_"
rm -f "$CHUNK_DIR/_rows.txt"
for f in "$CHUNK_DIR"/chunk_*.csv; do sed -i '1i stub' "$f"; done

n=0
for f in "$CHUNK_DIR"/chunk_*.csv; do
    # Retry qsub — the gadi PBS server occasionally refuses connections transiently.
    for attempt in 1 2 3 4 5; do
        jid=$(qsub -v chunk_csv="$f",gt_dir="$GT_DIR",out_base="$OUT_BASE",cover_threshold=10 \
              /home/147/cb8590/Projects/shelterbelts2/pbs_scripts/indices_eval.pbs 2>/dev/null)
        if [ -n "$jid" ]; then echo "Submitted $f ($(($(wc -l < "$f") - 1)) tiles) -> $jid"; n=$((n+1)); break; fi
        sleep 5
    done
done
echo "Submitted $n / $NCHUNKS jobs. Outputs -> $OUT_BASE/{e1,e2,e3,windmethod}"
