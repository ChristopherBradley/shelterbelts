#!/usr/bin/env python
"""
indices_batch_run.py — run the shelterbelt indices pipeline on eval tiles.

For each ground-truth percent-cover tile it runs 4 pipeline variants (each in
its own subprocess, so one bad tile can't kill the batch and memory can't
accumulate) and keeps only the outputs the shelter evaluation needs:

  windmethod (wind_method=ANY, large distances)  -> shelter_distances, shelter_categories
  default percentmethod, edge_size=1             -> linear_categories   (categories + Edge 1)
  default percentmethod, edge_size=2             -> linear_categories   (Edge 2)
  default percentmethod, edge_size=3             -> tree_categories     (Edge 3 + Core)

Ground-truth input is percent cover (0-100, 255 nodata); nodata is zeroed
before the pipeline so its `>= cover_threshold` tree mask is correct.
"""
import os
import re
import sys
import glob
import time
import shutil
import argparse
import subprocess

import pandas as pd
import rioxarray as rxr
import warnings
warnings.filterwarnings('ignore')

ALL_INDICES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'src', 'shelterbelts', 'classifications', '..', 'indices', 'all_indices.py')
ALL_INDICES = os.path.normpath(ALL_INDICES)
GT_DIR = '/scratch/xe2/cb8590/NSW_ag_treecover_10m'

# (variant name, extra CLI args, list of output layer names to keep)
VARIANTS = [
    ('windmethod', ['--wind_method', 'ANY', '--buffer_width', '5', '--max_shelterbelt_width', '7',
                    '--min_shelterbelt_length', '15', '--min_patch_size', '15', '--edge_size', '5',
                    '--min_core_size', '10000', '--wind_threshold', '15', '--crop_pixels', '20'],
     ['shelter_distances', 'shelter_categories']),
    ('e1', ['--edge_size', '1'], ['linear_categories']),
    ('e2', ['--edge_size', '2'], ['linear_categories']),
    ('e3', ['--edge_size', '3'], ['tree_categories', 'linear_categories']),
]


def prep_input(stub, tmp_in):
    src = os.path.join(GT_DIR, f'{stub}_percentcover.tif')
    if not os.path.exists(src):
        return None
    da = rxr.open_rasterio(src).isel(band=0).drop_vars('band')
    da = da.where(da != 255, 0).rio.write_nodata(None).astype('uint8')
    os.makedirs(tmp_in, exist_ok=True)
    out = os.path.join(tmp_in, f'{stub}.tif')
    da.rio.to_raster(out, compress='lzw')
    return out


def run_variant(tmp_in, tmp_out, tmpdir, cover_threshold, extra):
    os.makedirs(tmp_out, exist_ok=True)
    cmd = [sys.executable, ALL_INDICES, tmp_in, '--outdir', tmp_out, '--tmpdir', tmpdir,
           '--param_stub', 'eval', '--cover_threshold', str(cover_threshold), '--debug'] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)


def keep_outputs(stub, tmp_out, out_dir, layers):
    """Move the requested layers to out_dir/{stub}_{layer}.tif (handles the
    50-char stub truncation the pipeline applies to output filenames)."""
    os.makedirs(out_dir, exist_ok=True)
    moved = 0
    for layer in layers:
        matches = glob.glob(os.path.join(tmp_out, f'*_{layer}.tif'))
        if matches:
            shutil.move(matches[0], os.path.join(out_dir, f'{stub}_{layer}.tif'))
            moved += 1
    return moved


def done(stub, out_base):
    return (os.path.exists(f'{out_base}/windmethod/{stub}_shelter_distances.tif')
            and os.path.exists(f'{out_base}/e1/{stub}_linear_categories.tif')
            and os.path.exists(f'{out_base}/e2/{stub}_linear_categories.tif')
            and os.path.exists(f'{out_base}/e3/{stub}_tree_categories.tif'))


def main(chunk_csv, out_base, tmpdir, cover_threshold):
    stubs = pd.read_csv(chunk_csv)['stub'].tolist()
    n_ok = n_skip = n_fail = 0
    for stub in stubs:
        if done(stub, out_base):
            n_skip += 1
            continue
        t0 = time.time()
        work = os.path.join(tmpdir, f'work_{stub[:40]}')
        tmp_in = os.path.join(work, 'in')
        inp = prep_input(stub, tmp_in)
        if inp is None:
            n_fail += 1
            print(f'FAIL (no input): {stub}', flush=True)
            continue
        ok = True
        for name, extra, layers in VARIANTS:
            # Per-variant skip: don't recompute a variant whose outputs already
            # exist (lets a re-run redo only the variant that failed, e.g. windmethod).
            vdir = os.path.join(out_base, name)
            if all(os.path.exists(os.path.join(vdir, f'{stub}_{layer}.tif')) for layer in layers):
                continue
            tmp_out = os.path.join(work, name)
            r = run_variant(tmp_in, tmp_out, work, cover_threshold, extra)
            kept = keep_outputs(stub, tmp_out, os.path.join(out_base, name), layers)
            if kept < len(layers):
                # Run every variant independently: one failing variant must not
                # skip the others (e.g. a windmethod BARRA hiccup shouldn't drop
                # the percentmethod edge/category outputs).
                ok = False
                print(f'FAIL ({name}): {stub}\n{r.stderr[-1200:]}', flush=True)
        shutil.rmtree(work, ignore_errors=True)
        if ok and done(stub, out_base):
            n_ok += 1
            print(f'OK ({time.time()-t0:.0f}s): {stub}', flush=True)
        else:
            n_fail += 1
            print(f'PARTIAL/FAIL: {stub}', flush=True)
    print(f'\nChunk done: {n_ok} ok, {n_skip} skipped, {n_fail} failed', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('chunk_csv')
    ap.add_argument('--out_base', default='/scratch/xe2/cb8590/lidar_processing/eval_indices')
    ap.add_argument('--tmpdir', default='/scratch/xe2/cb8590/tmp/indices_work')
    ap.add_argument('--cover_threshold', type=int, default=10)
    ap.add_argument('--gt_dir', default=GT_DIR,
                    help='Folder of {stub}_percentcover.tif ground-truth tiles')
    args = ap.parse_args()
    GT_DIR = args.gt_dir
    os.makedirs(args.tmpdir, exist_ok=True)
    main(args.chunk_csv, args.out_base, args.tmpdir, args.cover_threshold)
