# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LAZ file inventory
#
# Scans all `.laz` files under a root directory and writes a GeoPackage with:
# - bbox polygon (reprojected to EPSG:4326)
# - point count
# - points per metre (ppm) derived from header bbox and point count
# - classification level parsed from filename (_C<n>_ segment)
# - whether the file has medium/high vegetation (classification >= 3)
# - acquisition year parsed from filename

# %%
import os
import re
import glob
import json
import argparse

import pdal
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS, Transformer


# %%
def extract_year(name):
    """Copied from analysis/demo_year_parsing.py."""
    if not isinstance(name, str):
        return None
    if name.startswith('Laura22021'):
        return 2021
    if name.startswith('Herbert1Lidar2020') or name.startswith('Herbert2Lidar2020'):
        return 2020
    m = re.search(r'20\d\d', name)
    return int(m.group()) if m else None


def _parse_classification(filename):
    """Return integer classification level from '-C<n>-' in filename, or None."""
    m = re.search(r'-C(\d+)-', os.path.basename(filename))
    return int(m.group(1)) if m else None


def _has_veg_from_classification(classification):
    """Return True if classification level implies vegetation classes (>= 3)."""
    if classification is None:
        return None
    return classification >= 3


def _read_header(laz_file):
    """Read bbox, CRS, and point count from LAZ header only (~0.04 s)."""
    try:
        p = pdal.Pipeline(json.dumps({
            "pipeline": [{"type": "readers.las", "filename": laz_file, "count": 0}]
        }))
        p.execute()
        m = p.metadata['metadata']['readers.las']
        result = {
            'filepath': laz_file,
            'minx': m['minx'], 'miny': m['miny'],
            'maxx': m['maxx'], 'maxy': m['maxy'],
            'wkt_crs': m['srs'].get('horizontal', '') if isinstance(m.get('srs'), dict) else '',
            'point_count': m['count'],
            'error': None,
        }
        del p
        return result
    except Exception as e:
        return {'filepath': laz_file, 'error': str(e)}


def _compute_ppm(point_count, minx, miny, maxx, maxy):
    """Points per square metre from header bbox."""
    area = (maxx - minx) * (maxy - miny)
    if area <= 0:
        return None
    return point_count / area


# %%
def build_inventory(laz_root, output_gpkg, limit=None):
    """Find all LAZ files, process them sequentially, save a GeoPackage."""
    all_files = sorted(glob.glob(os.path.join(laz_root, '**/*.laz'), recursive=True))
    print(f"Found {len(all_files)} laz files under {laz_root}")

    if limit is not None:
        all_files = all_files[:limit]
        print(f"Limiting to {len(all_files)} files")

    rows = []
    for i, laz_file in enumerate(all_files, 1):
        result = _read_header(laz_file)
        if result.get('error'):
            print(f"  [{i}/{len(all_files)}] ERROR {laz_file}: {result['error']}")
            continue

        name = os.path.basename(laz_file)
        classification = _parse_classification(laz_file)
        has_veg = _has_veg_from_classification(classification)
        ppm = _compute_ppm(result['point_count'], result['minx'], result['miny'], result['maxx'], result['maxy'])
        year = extract_year(name)

        try:
            geom = box(result['minx'], result['miny'], result['maxx'], result['maxy'])
            if result['wkt_crs']:
                src_crs = CRS.from_wkt(result['wkt_crs'])
                transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
                minx, miny = transformer.transform(result['minx'], result['miny'])
                maxx, maxy = transformer.transform(result['maxx'], result['maxy'])
                geom = box(minx, miny, maxx, maxy)
        except Exception:
            geom = box(result['minx'], result['miny'], result['maxx'], result['maxy'])

        rows.append({
            'filepath': laz_file,
            'year': year,
            'classification': classification,
            'has_veg': has_veg,
            'point_count': result['point_count'],
            'ppm': ppm,
            'geometry': geom,
        })

        if i % 100 == 0 or i == len(all_files):
            print(f"  {i}/{len(all_files)} files processed", flush=True)

    gdf = gpd.GeoDataFrame(rows, crs='EPSG:4326')
    gdf.to_file(output_gpkg, driver='GPKG')
    print(f"Saved: {output_gpkg}  ({len(gdf)} tiles)")


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a GeoPackage inventory of LAZ files")
    parser.add_argument('laz_root', help="Root directory to search for .laz files")
    parser.add_argument('output_gpkg', nargs='?', help="Path for the output GeoPackage (default: laz_root + .gpkg)")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N files")
    args = parser.parse_args()

    if args.output_gpkg is None:
        args.output_gpkg = args.laz_root.rstrip('/') + '.gpkg'

    build_inventory(args.laz_root, args.output_gpkg, limit=args.limit)
