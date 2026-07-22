"""Crop percent-cover ground truth for the adjacent evaluation tiles.

The adjacent tiles (one diagonal neighbour per Nick-matched tile, from
build_adjacent_eval_tiles.py) fall inside the already-processed 30 km region
mosaics. This crops each tile's window straight out of its covering region tif
in the region's NATIVE UTM CRS — no reprojection, so no pixel offset. Writes one
{laz_stub}_percentcover.tif per tile plus a footprints gpkg (stub + koppen).

Caveat: the region mosaics were built before photogrammetry tiles were removed,
so some pixels may be photogrammetry-derived. Tiles whose footprint overlaps a
PHO source tile are flagged `pho_overlap` in the footprints gpkg.
"""
import os
import re
import glob
import argparse

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

ADJ_GPKG = '/scratch/xe2/cb8590/Adjacent_eval_tiles/adjacent_eval_tiles.gpkg'
REGION_GLOB = '/scratch/xe2/cb8590/lidar_processing/regions/*percentcover_res10_height2m.tif'
LIDAR = '/g/data/xe2/cb8590/Outlines_Lidar/lidar_recent.gpkg'
KOPPEN = '/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned2.gpkg'
OUT_DIR = '/scratch/xe2/cb8590/Adjacent_eval_tiles'
FOOT_OUT = f'{OUT_DIR}/adjacent_eval_footprints.gpkg'

_PRODUCT_RE = re.compile(r'-([A-Za-z]+\d)-C\d+-')
MIN_VALID_FRAC = 0.6


def region_index():
    recs = []
    for p in glob.glob(REGION_GLOB):
        with rasterio.open(p) as s:
            recs.append({'path': p, 'crs': str(s.crs),
                         'geometry': gpd.GeoSeries([box(*s.bounds)], crs=s.crs).to_crs(4326).iloc[0]})
    return gpd.GeoDataFrame(recs, crs=4326)


def pho_union_4326():
    g = gpd.read_file(LIDAR)
    g['product'] = g['filepath'].apply(
        lambda p: (_PRODUCT_RE.search(os.path.basename(p)).group(1)
                   if _PRODUCT_RE.search(os.path.basename(p)) else None))
    pho = g[g['product'].fillna('').str.startswith('PHO')].to_crs(4326)
    return pho.union_all() if len(pho) else None


def crop_one(tile_geom_4326, ridx):
    """Crop the tile window from its best-overlap region tif (native CRS).
    Mosaic same-CRS regions where the tile straddles two 30 km blocks."""
    hits = ridx[ridx.intersects(tile_geom_4326)].copy()
    if hits.empty:
        return None, None, None, 0.0
    hits['ov'] = hits.geometry.intersection(tile_geom_4326).area
    hits = hits.sort_values('ov', ascending=False)
    crs = hits.iloc[0]['crs']
    hits = hits[hits['crs'] == crs]                       # keep one UTM zone

    tb = gpd.GeoSeries([tile_geom_4326], crs=4326).to_crs(crs).total_bounds
    data = transform = profile = None
    for _, h in hits.iterrows():
        with rasterio.open(h['path']) as src:
            win = from_bounds(*tb, src.transform)
            d = src.read(1, window=win, boundless=True, fill_value=255)
            if d.size == 0 or d.shape[0] == 0 or d.shape[1] == 0:
                continue
            t = src.window_transform(win)
            if data is None:
                data, transform, profile = d, t, src.profile.copy()
            else:
                data = np.where(data == 255, d, data)     # fill gaps from the next block
    if data is None:
        return None, None, None, 0.0
    vf = float((data != 255).mean())
    profile.update(height=data.shape[0], width=data.shape[1], transform=transform,
                   count=1, dtype='uint8', nodata=255, compress='lzw', crs=crs)
    return data, profile, crs, vf


def main(tiles_gpkg, stub_col, strip_suffix, out_dir, foot_out):
    os.makedirs(out_dir, exist_ok=True)
    adj = gpd.read_file(tiles_gpkg).to_crs(4326)
    ridx = region_index()
    pho = pho_union_4326()
    koppen = gpd.read_file(KOPPEN).to_crs(4326)
    print(f'tiles: {len(adj)}   region tifs: {len(ridx)}', flush=True)

    rows, n_ok, n_bad = [], 0, {'no_region': 0, 'low_coverage': 0}
    for i, t in adj.iterrows():
        stub = str(t[stub_col]).replace(strip_suffix, '') if strip_suffix else t[stub_col]
        data, profile, crs, vf = crop_one(t.geometry, ridx)
        if data is None:
            n_bad['no_region'] += 1
            continue
        if vf < MIN_VALID_FRAC:
            n_bad['low_coverage'] += 1
            continue
        out = f'{out_dir}/{stub}_percentcover.tif'
        with rasterio.open(out, 'w', **profile) as dst:
            dst.write(data, 1)
        n_ok += 1
        rows.append({'stub': stub, 'source_stub': t.get('source_stub', ''),
                     'valid_frac': round(vf, 3),
                     'overlap_any_nick': t.get('overlap_any_nick', np.nan),
                     'pho_overlap': bool(pho is not None and t.geometry.intersects(pho)),
                     'geometry': t.geometry})
        if n_ok % 50 == 0:
            print(f'  cropped {n_ok}', flush=True)

    foot = gpd.GeoDataFrame(rows, crs=4326)
    foot = gpd.sjoin(foot, koppen[['Name', 'geometry']], how='left', predicate='intersects') \
        .drop_duplicates('stub').drop(columns='index_right', errors='ignore').rename(columns={'Name': 'koppen'})
    foot.to_file(foot_out, driver='GPKG')
    print(f'\ncropped OK: {n_ok}   skipped: {n_bad}', flush=True)
    print(f'  tiles overlapping photogrammetry source: {int(foot["pho_overlap"].sum())}', flush=True)
    print(f'saved percentcover tiles + {foot_out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles_gpkg', default=ADJ_GPKG)
    ap.add_argument('--stub_col', default='laz_stub')
    ap.add_argument('--strip_suffix', default='')
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--foot_out', default=FOOT_OUT)
    args = ap.parse_args()
    main(args.tiles_gpkg, args.stub_col, args.strip_suffix, args.out_dir, args.foot_out)
