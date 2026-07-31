#!/usr/bin/env python
"""
crop_comparison_rasters.py — crop GCH v1, GCH v2 and WorldCover to each eval tile.

Claude-generated code - evaluated by viewing the resulting tifs in QGIS alongside the lidar-derived tifs.
"""
import os
import argparse

import geopandas as gpd
import rioxarray as rxr
from rioxarray.exceptions import NoDataInBounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import warnings
warnings.filterwarnings('ignore')

EVAL_FOOTPRINTS = '/scratch/xe2/cb8590/NSW_ag_treecover_10m/eval_dataset_footprints.gpkg'
GT_DIR = '/scratch/xe2/cb8590/NSW_ag_treecover_10m'

GCH1_VRT = '/scratch/xe2/cb8590/tmp/vrts/gch_v1.vrt'
GCH2_VRT = '/scratch/xe2/cb8590/tmp/vrts/gch_v2.vrt'
WC_VRT = '/scratch/xe2/cb8590/tmp/vrts/worldcover.vrt'


def build_sources(gch1_dir, gch2_dir, wc_dir):
    return {
        'canopy_height':    (GCH1_VRT, gch1_dir, Resampling.max),
        'canopy_height_v2': (GCH2_VRT, gch2_dir, Resampling.max),
        'worldcover':       (WC_VRT,   wc_dir,   Resampling.nearest),
    }


def crop_source(ref, vrt_path, resampling):
    """Clip a source VRT to ref's bounds (windowed read) and reproject-match to ref."""
    src = rxr.open_rasterio(vrt_path).isel(band=0).drop_vars('band')
    minx, miny, maxx, maxy = ref.rio.bounds()
    # Transform ref bounds into the source CRS, pad by ~2 pixels of the source.
    sxmin, symin, sxmax, symax = transform_bounds(ref.rio.crs, src.rio.crs, minx, miny, maxx, maxy)
    px = abs(src.rio.resolution()[0]) * 3
    clipped = src.rio.clip_box(sxmin - px, symin - px, sxmax + px, symax + px)
    return clipped.rio.reproject_match(ref, resampling=resampling)


def main(footprints, gt_dir, sources, limit):
    for _, (_, out_dir, _) in sources.items():
        os.makedirs(out_dir, exist_ok=True)

    gdf = gpd.read_file(footprints)
    if limit:
        gdf = gdf.iloc[:limit]
    print(f'cropping {len(gdf)} tiles x {len(sources)} sources', flush=True)

    counts = {k: 0 for k in sources}
    for i, row in gdf.iterrows():
        stub = row['stub']
        ref_path = os.path.join(gt_dir, f'{stub}_percentcover.tif')
        ref = rxr.open_rasterio(ref_path).isel(band=0).drop_vars('band')
        for name, (vrt, out_dir, resampling) in sources.items():
            # v2 keeps the same `_canopy_height` filename suffix (separate folder)
            suffix = 'canopy_height' if name == 'canopy_height_v2' else name
            out_path = os.path.join(out_dir, f'{stub}_{suffix}.tif')
            if os.path.exists(out_path):
                counts[name] += 1
                continue
            try:
                da = crop_source(ref, vrt, resampling)
                da.rio.to_raster(out_path, compress='lzw')
                counts[name] += 1
            except NoDataInBounds:
                pass  # source doesn't cover this tile (shouldn't happen in NSW)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(gdf)}  {counts}', flush=True)
    print(f'done: {counts}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--footprints', default=EVAL_FOOTPRINTS)
    ap.add_argument('--gt_dir', default=GT_DIR)
    ap.add_argument('--gch1_dir', default='/scratch/xe2/cb8590/NSW_ag_GCH')
    ap.add_argument('--gch2_dir', default='/scratch/xe2/cb8590/NSW_ag_GCH_v2')
    ap.add_argument('--wc_dir', default='/scratch/xe2/cb8590/NSW_ag_worldcover')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    sources = build_sources(args.gch1_dir, args.gch2_dir, args.wc_dir)
    main(args.footprints, args.gt_dir, sources, args.limit)
