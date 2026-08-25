"""Mask merged region rasters to the agricultural area, setting everything outside -> 255.

Two masks are applied, and a pixel must pass both:

  ag tiles : the merged region rasters are 2 degree rectangles; rasterio fills the gaps between the
             scattered ag tiles with 0, which collides with the real 'Not Trees' (0) category and
             inflates value-0 in sparse regions, so we rasterize the region's ag 4 km tiles.
  NLUM     : the ag 4 km tiles are a *tile-level* footprint (a whole 4 km tile is kept if it holds
             even one valid NLUM 250 m pixel), so on their own they cover ~3.06e10 10 m pixels --
             2.6x the ~1.18e10 of the old predicted-tree run. Sampling NLUM per pixel brings it back
             to the same ag area, which is what makes the totals comparable across runs.

    python gch_mask.py --start 0 --end 20      # mask regions [0:20] of the sorted region list
"""
import os
import argparse

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from gch_prep import region_tiles

INDICES_DIR = '/scratch/xe2/cb8590/gch_v2_ag_indices/indices'
OUT_DIR = '/scratch/xe2/cb8590/gch_v2_ag_indices/indices_masked'
AG_BBOXS = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_ag.gpkg'
NLUM_TIF = '/g/data/xe2/cb8590/Outlines/NLUM_v7_probSurf_2021_331_5_W_CER.tif'
DATATYPES = ['shelter_categories', 'shelter_distances', 'shelter_densities', 'opportunities']

R_MERC = 6378137.0        # EPSG:3857 sphere radius


def all_regions(bboxs_gdf):
    import math
    c = bboxs_gdf.geometry.centroid
    return sorted(set(f'lat_{int(math.floor(abs(y)/2)*2)}_lon_{int(math.floor(x/2)*2)}'
                      for y, x in zip(c.y.values, c.x.values)))


def nlum_valid(transform, H, W, nlum_tif=NLUM_TIF, block_rows=2048):
    """Boolean (H, W) mask: True where the EPSG:3857 pixel centre lands on a non-nodata NLUM pixel.

    Web Mercator is separable, so the pixel-centre lon depends only on the column and the lat only
    on the row -- no per-pixel reprojection needed, just two 1-D lookups into the NLUM grid. NLUM is
    GDA94 and we treat the inverse-Mercator output as WGS84; the ~1 m datum offset is irrelevant
    against 250 m pixels. Filled in row blocks to keep peak memory near the output array itself.
    """
    x = transform.c + (np.arange(W) + 0.5) * transform.a
    y = transform.f + (np.arange(H) + 0.5) * transform.e
    lon = np.degrees(x / R_MERC)
    lat = np.degrees(2 * np.arctan(np.exp(y / R_MERC)) - np.pi / 2)

    out = np.zeros((H, W), dtype=bool)
    with rasterio.open(nlum_tif) as n:
        nt, nodata = n.transform, n.nodata
        cc = np.floor((lon - nt.c) / nt.a).astype(np.int64)
        rr = np.floor((lat - nt.f) / nt.e).astype(np.int64)
        ci = np.flatnonzero((cc >= 0) & (cc < n.width))
        ri = np.flatnonzero((rr >= 0) & (rr < n.height))
        if ci.size == 0 or ri.size == 0:
            return out                                # region lies entirely off the NLUM grid
        c0, c1 = cc[ci].min(), cc[ci].max() + 1
        r0, r1 = rr[ri].min(), rr[ri].max() + 1
        valid = n.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)) != nodata

    cols = cc[ci] - c0
    for b in range(0, ri.size, block_rows):
        rows_i = ri[b:b + block_rows]
        out[rows_i[0]:rows_i[-1] + 1, ci[0]:ci[-1] + 1] = valid[np.ix_(rr[rows_i] - r0, cols)]
    return out


def mask_region(region, bboxs_gdf, indices_dir=INDICES_DIR, out_dir=OUT_DIR, datatypes=DATATYPES,
                nlum_tif=NLUM_TIF):
    ref = os.path.join(indices_dir, f'{region}_merged_shelter_categories.tif')
    if not os.path.exists(ref):
        print(f"  [{region}] no merged raster, skip", flush=True)
        return 0
    sub, _ = region_tiles(bboxs_gdf, region)          # ag 4 km tiles in EPSG:3857
    with rasterio.open(ref) as s:
        transform, H, W, crs = s.transform, s.height, s.width, s.crs
    inside = rasterize([(g, 1) for g in sub.geometry], out_shape=(H, W), transform=transform,
                       fill=0, dtype='uint8') == 1
    if nlum_tif:
        inside &= nlum_valid(transform, H, W, nlum_tif)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for d in datatypes:
        f = os.path.join(indices_dir, f'{region}_merged_{d}.tif')
        if not os.path.exists(f):
            continue
        with rasterio.open(f) as s:
            a = s.read(1)
            prof = s.profile
        a[~inside] = 255
        prof.update(nodata=255, compress='deflate')
        with rasterio.open(os.path.join(out_dir, f'{region}_merged_{d}.tif'), 'w', **prof) as dst:
            dst.write(a, 1)
        n += 1
    print(f"  [{region}] masked {n} rasters", flush=True)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=None)
    p.add_argument('--region', default=None, help='Mask a single region instead of a slice')
    p.add_argument('--indices_dir', default=INDICES_DIR)
    p.add_argument('--out_dir', default=OUT_DIR)
    p.add_argument('--nlum_tif', default=NLUM_TIF,
                   help="NLUM raster whose non-nodata pixels define the ag area ('' to skip)")
    args = p.parse_args()

    bboxs = gpd.read_file(AG_BBOXS).set_crs(4326, allow_override=True)
    regions = [args.region] if args.region else all_regions(bboxs)[args.start:args.end]
    print(f"Masking {len(regions)} regions -> {args.out_dir}", flush=True)
    print(f"NLUM pixel mask: {args.nlum_tif or 'DISABLED (ag tiles only)'}", flush=True)
    for r in regions:
        mask_region(r, bboxs, args.indices_dir, args.out_dir, nlum_tif=args.nlum_tif)


if __name__ == '__main__':
    main()
