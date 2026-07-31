#!/usr/bin/env python
"""
Run lidar.py on every tile in a chunk gpkg, one subprocess per tile.

Skips tiles where the expected output already exists.
"""
# Should maybe move this into the main lidar.py 
import os
import re
import sys
import glob
import time
import argparse
import subprocess

import geopandas as gpd

LIDAR_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'src', 'shelterbelts', 'classifications', 'lidar.py')

# MGA zone token in the filename ("_55_", "_56_", "_54_")
_ZONE_RE = re.compile(r'_(5[456])_')

def epsg_from_filename(laz_file):
    m = _ZONE_RE.search(os.path.basename(laz_file))
    return f'283{m.group(1)}' if m else None


def already_done(outdir, stub):
    """The final expected output (uint8 crown-masked CHM) exists for this stub."""
    return len(glob.glob(os.path.join(outdir, f'{stub}_chm_crowns_res1_uint8.tif'))) > 0


def run_chunk(chunk_gpkg, outdir, dem, column='filepath', timeout=1800):
    os.makedirs(outdir, exist_ok=True)
    gdf = gpd.read_file(chunk_gpkg)
    log_path = os.path.join(outdir, os.path.basename(chunk_gpkg).replace('.gpkg', '_status.csv'))

    n_ok = n_skip = n_fail = 0
    with open(log_path, 'w') as log:
        log.write('stub,status,seconds\n')
        for laz_file in gdf[column]:
            stub = os.path.basename(laz_file).rsplit('.', 1)[0]

            if already_done(outdir, stub):
                n_skip += 1
                log.write(f'{stub},skip,0\n')
                continue

            if not os.path.exists(laz_file) or os.path.getsize(laz_file) == 0:
                n_fail += 1
                log.write(f'{stub},missing_or_empty,0\n')
                continue

            cmd = [sys.executable, LIDAR_PY, laz_file,
                   '--outdir', outdir, '--stub', stub, '--dem', dem,
                   '--delineate_crowns', '--uint8']
            epsg = epsg_from_filename(laz_file)
            if epsg:
                cmd += ['--epsg', epsg]

            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            elapsed = time.time() - t0

            if result.returncode == 0 and already_done(outdir, stub):
                n_ok += 1
                log.write(f'{stub},ok,{elapsed:.1f}\n')
                print(f"OK   ({elapsed:.1f}s): {stub}", flush=True)
            else:
                n_fail += 1
                log.write(f'{stub},fail,{elapsed:.1f}\n')
                print(f"FAIL ({elapsed:.1f}s): {stub}", flush=True)
                print(result.stderr[-2000:], flush=True)

    print(f"\nChunk done: {n_ok} ok, {n_skip} skipped (already done), {n_fail} failed",
          flush=True)
    return n_ok, n_skip, n_fail


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run lidar.py on every tile in a chunk gpkg, one subprocess per tile.")
    parser.add_argument('chunk_gpkg', help="GeoPackage listing laz tiles for this job")
    parser.add_argument('--outdir', required=True, help="Output directory")
    parser.add_argument('--dem', required=True, help="DEM folder passed through to lidar.py")
    parser.add_argument('--column', default='filepath', help="Column with laz paths")
    parser.add_argument('--timeout', type=int, default=1800, help="Per-tile timeout in seconds")
    args = parser.parse_args()

    run_chunk(args.chunk_gpkg, args.outdir, args.dem, args.column, args.timeout)
