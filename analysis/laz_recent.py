import os
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd


# Keep the tile with the most recent acquisition if footprints overlap
def _recency_keys(gdf):
    """Return a list of comparable keys; a larger key means 'keep this one'."""
    year = gdf['year'].fillna(-1).to_numpy()
    ppm = gdf['ppm'].fillna(-1).to_numpy() if 'ppm' in gdf else np.full(len(gdf), -1.0)
    pc = gdf['point_count'].fillna(-1).to_numpy() if 'point_count' in gdf else np.full(len(gdf), -1)
    names = gdf['_basename'].to_numpy()
    paths = gdf['filepath'].to_numpy()
    return list(zip(year, ppm, pc, names, paths))


def laz_recent(gpkg_path, overlap_threshold=0.9, limit=None):
    gdf = gpd.read_file(gpkg_path)
    if limit is not None:
        gdf = gdf.iloc[:limit]
    gdf = gdf.reset_index(drop=True)

    n_original = len(gdf)
    print(f"Total tiles: {n_original}", flush=True)

    gdf['_basename'] = gdf['filepath'].apply(os.path.basename)

    # Every geometry is an axis-aligned bbox, so overlaps are exact arithmetic
    # on the bounds (minx, miny, maxx, maxy)
    bounds = gdf.geometry.bounds.to_numpy()
    widths = bounds[:, 2] - bounds[:, 0]
    heights = bounds[:, 3] - bounds[:, 1]
    areas = widths * heights
    keys = _recency_keys(gdf)

    sindex = gdf.sindex
    to_remove = set()

    for i in range(n_original):
        if i in to_remove:
            continue
        minx_i, miny_i, maxx_i, maxy_i = bounds[i]
        area_i = areas[i]
        if area_i <= 0:
            continue
        key_i = keys[i]
        for j in sindex.intersection((minx_i, miny_i, maxx_i, maxy_i)):
            if j == i or j in to_remove:
                continue
            # Only a strictly-newer neighbour can evict tile i.
            if keys[j] <= key_i:
                continue
            # Analytic overlap of two axis-aligned boxes.
            ix = min(maxx_i, bounds[j, 2]) - max(minx_i, bounds[j, 0])
            iy = min(maxy_i, bounds[j, 3]) - max(miny_i, bounds[j, 1])
            if ix <= 0 or iy <= 0:
                continue
            if (ix * iy) / area_i > overlap_threshold:
                to_remove.add(i)
                break

    n_removed = len(to_remove)
    n_kept = n_original - n_removed

    total_gb = float(gdf['filesize_gb'].sum())
    removed_gb = float(gdf.loc[list(to_remove), 'filesize_gb'].sum()) if to_remove else 0.0
    kept_gb = total_gb - removed_gb
    print(f"Kept:    {n_kept}  ({kept_gb / 1000:.2f} TB)", flush=True)
    print(f"Removed: {n_removed}  ({removed_gb / 1000:.2f} TB)", flush=True)
    print(f"Total:   {n_original}  ({total_gb / 1000:.2f} TB)", flush=True)

    gdf_recent = gdf.drop(index=list(to_remove)).drop(columns=['_basename'])

    stem, ext = os.path.splitext(gpkg_path)
    out_path = stem + '_recent' + ext
    gdf_recent.to_file(out_path, driver='GPKG')
    print(f"Saved: {out_path}", flush=True)
    return gdf_recent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter a LAZ tile GeoPackage to keep only the most recent tile where tiles overlap."
    )
    parser.add_argument('gpkg', help="Path to the input GeoPackage")
    parser.add_argument('--overlap_threshold', type=float, default=0.9,
                        help="Min fraction of a tile covered by a newer tile to drop it")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only examine the first N rows")
    args = parser.parse_args()

    laz_recent(args.gpkg, overlap_threshold=args.overlap_threshold, limit=args.limit)
