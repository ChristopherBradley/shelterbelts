# Prep: list the ACT2020 laz tiles overlapping snakes_bbox.gpkg, for parallel
# per-tile PBS processing (chm + crown delineation), merged in serial afterwards.
#
# Only ACT2020 is used (12ppm, higher quality; it fully covers the bbox). The NSW
# /scratch/xe2/cb8590/lidar tiles overlap it but are deliberately excluded.

import os
import re
import glob

import geopandas as gpd
from shapely.geometry import box

BBOX = '/home/147/cb8590/Projects/shelterbelts/outdir/snakes_bbox.gpkg'
ACT_DIR = '/scratch/xe2/cb8590/elvis-shelterbelts/ACT2020'
WORKDIR = '/scratch/xe2/cb8590/lidar_processing/snakes'
ACT_CRS = 'EPSG:7855'   # GDA2020 / MGA zone 55, from the ACT2020 laz headers


def act_tiles_intersecting(bbox_path=BBOX, act_dir=ACT_DIR):
    """Every ACT2020 laz whose 1km tile footprint (parsed from its name) hits the bbox."""
    bbox = gpd.read_file(bbox_path).to_crs(ACT_CRS).geometry.iloc[0]
    paths = []
    for f in sorted(glob.glob(os.path.join(act_dir, '*.laz'))):
        m = re.search(r'AHD_(\d{3})(\d{4})_', os.path.basename(f))
        if not m:
            continue
        e, n = int(m.group(1)) * 1000, int(m.group(2)) * 1000
        if box(e, n, e + 1000, n + 1000).intersects(bbox):
            paths.append(f)
    return paths


if __name__ == '__main__':
    os.makedirs(WORKDIR, exist_ok=True)
    paths = act_tiles_intersecting()
    listfile = os.path.join(WORKDIR, 'snakes_act_laz.txt')
    with open(listfile, 'w') as fh:
        fh.write('\n'.join(paths) + '\n')
    print(f"Found {len(paths)} ACT2020 tiles intersecting the bbox")
    print(f"Wrote laz list to {listfile}")
