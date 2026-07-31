# Serial merge step for the snakes_bbox lidar run.
#
# After snakes_bbox.pbs has produced a per-tile 1m CHM (masked to crowns) and a
# crowns gpkg for every ACT2020 tile, this merges them into a single CHM tif and
# a single crowns gpkg, then clips both to snakes_bbox.gpkg.

import os
import glob
import geopandas as gpd
import rioxarray as rxr

from shelterbelts.classifications.merge_tifs import merge_tifs
from shelterbelts.classifications.merge_gpkg import merge_gpkg

CHMS_DIR = '/scratch/xe2/cb8590/lidar_processing/snakes/chms'
TMPDIR = '/scratch/xe2/cb8590/lidar_processing/snakes/tmp'
BBOX = '/home/147/cb8590/Projects/shelterbelts/outdir/snakes_bbox.gpkg'
OUTDIR = '/home/147/cb8590/Projects/shelterbelts/outdir'


def main():
    os.makedirs(TMPDIR, exist_ok=True)
    bbox = gpd.read_file(BBOX)

    n_tif = len(glob.glob(os.path.join(CHMS_DIR, '*_chm_crowns_*.tif')))
    n_gpkg = len(glob.glob(os.path.join(CHMS_DIR, '*_crowns.gpkg')))
    print(f"Merging {n_tif} CHM tifs and {n_gpkg} crown gpkgs from {CHMS_DIR}")

    # --- crowns gpkg first (before merge_tifs writes a *_footprints.gpkg here) ---
    merged_crowns = merge_gpkg(CHMS_DIR, suffix='_crowns.gpkg')
    crowns_clip = merged_crowns.clip(bbox.to_crs(merged_crowns.crs))
    crowns_out = os.path.join(OUTDIR, 'snakes_crowns.gpkg')
    crowns_clip.to_file(crowns_out, driver='GPKG')
    print(f"Saved {len(crowns_clip)} crowns (clipped to bbox) to: {crowns_out}")

    # --- CHM mosaic (all tiles already uint8 in EPSG:7855) ---
    da = merge_tifs(CHMS_DIR, tmpdir=TMPDIR, suffix='_uint8.tif', dont_reproject=False)
    # da is reprojected to the estimated WGS84 UTM CRS (EPSG:32755, same as the bbox)
    b = bbox.to_crs(da.rio.crs).total_bounds
    chm_clip = da.rio.clip_box(minx=b[0], miny=b[1], maxx=b[2], maxy=b[3])
    chm_out = os.path.join(OUTDIR, 'snakes_chm.tif')
    chm_clip.rio.to_raster(chm_out, compress='lzw')
    print(f"Saved CHM (clipped to bbox) to: {chm_out}")


if __name__ == '__main__':
    main()
