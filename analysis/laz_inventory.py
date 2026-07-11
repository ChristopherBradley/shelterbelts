"""LAZ file inventory.

Scans all `.laz` files under a root directory and writes a GeoPackage with:
- bbox polygon (reprojected to EPSG:4326)
- point count
- points per square metre (ppm) derived from header bbox and point count
- classification level parsed from filename (-C<n>- or _C<n>_ segment)
- MGA zone parsed from filename
- acquisition year parsed from filename

"""
import os
import re
import glob
import struct
import argparse
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS, Transformer

# The MGA zone token: a standalone 54/55/56 bounded by underscores. Tile ids are
# 6-7 digits so they don't collide with this 2-digit pattern.
_ZONE_RE = re.compile(r'_(5[456])_')

# LAS public header block field offsets (little-endian), common to LAS 1.2-1.4.
_OFF_VERSION_MAJOR = 24      # uint8
_OFF_LEGACY_COUNT = 107      # uint32 (legacy point count, LAS < 1.4)
_OFF_COUNT_1_4 = 247         # uint64 (point count, LAS 1.4+)
_OFF_BBOX = 179              # 4 x float64: MaxX, MinX, MaxY, MinY
_HEADER_BYTES = 383          # enough to cover every field above


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
    """Integer classification level from '-C<n>-' or '_C<n>_' in filename."""
    m = re.search(r'[-_]C(\d+)[-_]', os.path.basename(filename))
    return int(m.group(1)) if m else None


def _parse_mga_zone(filename):
    """MGA zone (54/55/56) from the filename, or None if absent."""
    m = _ZONE_RE.search(os.path.basename(filename))
    return int(m.group(1)) if m else None


def _read_header(laz_file):
    """Read bbox and point count straight from the LAS header."""
    try:
        with open(laz_file, 'rb') as fh:
            buf = fh.read(_HEADER_BYTES)
        if buf[:4] != b'LASF':
            return {'filepath': laz_file, 'error': 'not a LAS/LAZ file'}
        vmajor, vminor = buf[_OFF_VERSION_MAJOR], buf[_OFF_VERSION_MAJOR + 1]
        legacy = struct.unpack_from('<I', buf, _OFF_LEGACY_COUNT)[0]
        count_1_4 = struct.unpack_from('<Q', buf, _OFF_COUNT_1_4)[0]
        count = count_1_4 if ((vmajor, vminor) >= (1, 4) and count_1_4 > 0) else legacy
        maxx, minx, maxy, miny = struct.unpack_from('<4d', buf, _OFF_BBOX)
        return {
            'filepath': laz_file,
            'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
            'point_count': count,
            'error': None,
        }
    except Exception as e:
        return {'filepath': laz_file, 'error': str(e)}


def _compute_ppm(point_count, minx, miny, maxx, maxy):
    """Points per square metre from header bbox."""
    area = (maxx - minx) * (maxy - miny)
    if area <= 0:
        return None
    return point_count / area


def build_inventory(laz_root, output_gpkg, limit=None, nthreads=32):
    """Find all LAZ files, read headers in parallel, save a GeoPackage."""
    all_files = sorted(glob.glob(os.path.join(laz_root, '**/*.laz'), recursive=True))
    print(f"Found {len(all_files)} laz files under {laz_root}", flush=True)

    if limit is not None:
        all_files = all_files[:limit]
        print(f"Limiting to {len(all_files)} files", flush=True)

    # I/O-bound header reads: overlap them across threads (stays on one CPU).
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        headers = list(ex.map(_read_header, all_files))

    # One reprojection Transformer per MGA zone (there are only 54/55/56), reused
    # for every tile in that zone.
    transformers = {}

    def _transformer(zone):
        if zone not in transformers:
            transformers[zone] = Transformer.from_crs(
                CRS.from_epsg(28300 + zone), CRS.from_epsg(4326), always_xy=True)
        return transformers[zone]

    rows = []
    errors = []
    for h in headers:
        laz_file = h['filepath']
        if h.get('error'):
            errors.append((laz_file, h['error']))
            continue

        name = os.path.basename(laz_file)
        zone = _parse_mga_zone(laz_file)

        if zone is not None:
            # Reproject all four corners and take the envelope (a UTM box is not
            # axis-aligned in 4326, so transforming only two corners clips it).
            tr = _transformer(zone)
            xs, ys = tr.transform(
                [h['minx'], h['maxx'], h['minx'], h['maxx']],
                [h['miny'], h['miny'], h['maxy'], h['maxy']])
            geom = box(min(xs), min(ys), max(xs), max(ys))
        else:
            # No zone in filename (not seen in the NSW data) — keep raw coords.
            geom = box(h['minx'], h['miny'], h['maxx'], h['maxy'])

        rows.append({
            'filepath': laz_file,
            'year': extract_year(name),
            'classification_level': _parse_classification(laz_file),
            'mga_zone': zone,
            'point_count': h['point_count'],
            'ppm': _compute_ppm(h['point_count'], h['minx'], h['miny'],
                                h['maxx'], h['maxy']),
            'filesize_gb': os.path.getsize(laz_file) / 1e9,
            'geometry': geom,
        })

    if errors:
        print(f"WARNING: {len(errors)} files failed to read:", flush=True)
        for laz_file, err in errors[:20]:
            print(f"  {laz_file}: {err}", flush=True)

    gdf = gpd.GeoDataFrame(rows, crs='EPSG:4326')
    gdf.to_file(output_gpkg, driver='GPKG')
    print(f"Saved: {output_gpkg}  ({len(gdf)} tiles, {len(errors)} errors)",
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a GeoPackage inventory of LAZ files")
    parser.add_argument('laz_root', help="Root directory to search for .laz files")
    parser.add_argument('output_gpkg', nargs='?', help="Path for the output GeoPackage (default: laz_root + .gpkg)")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N files")
    parser.add_argument('--nthreads', type=int, default=32, help="Threads for header reads (I/O-bound)")
    args = parser.parse_args()

    if args.output_gpkg is None:
        args.output_gpkg = args.laz_root.rstrip('/') + '.gpkg'

    build_inventory(args.laz_root, args.output_gpkg, limit=args.limit, nthreads=args.nthreads)
