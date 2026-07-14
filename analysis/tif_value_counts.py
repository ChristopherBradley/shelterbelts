#!/usr/bin/env python
"""
tif_value_counts.py — pixel value_counts across many GeoTIFFs, at scale.

Counts how many pixels take each value across a set of uint8 rasters (values
0-100, nodata 255) — like a pd.value_counts, but aggregated over many tifs.
Optionally splits the counts by spatial zone: supply a polygon file (gpkg/shp)
and a column to group by, and pixels are counted within each polygon/zone.

Memory safety: each raster is streamed in row-chunks so peak memory stays low
regardless of raster size, and the ONLY thing that accumulates across tifs are
the small length-256 count vectors (one per zone). Nothing holds onto raster
arrays, so there is no memory growth as the number of tifs increases. When
zones are used, the zone label raster is also built one window at a time, so it
never materialises a full-tile label array either.

Usage:
    # Whole folder, global counts only (cheapest):
    python tif_value_counts.py --tif_folder <dir> --suffix <suffix> \
        --output_csv <csv>

    # Split by polygon zones (e.g. Koppen regions / state boundaries):
    python tif_value_counts.py --tif_folder <dir> --suffix <suffix> \
        --zones_file <regions.gpkg> --zone_col Name --output_csv <csv>

    # A single tif:
    python tif_value_counts.py --tif <file.tif> --output_csv <csv>

Output CSV columns:
    zone, value, label, count, percent, is_nodata
The special zone "ALL" holds the global (whole-dataset) counts. When no zones
file is given, only the "ALL" zone is written. `percent` is percent of valid
(non-nodata) pixels within that zone.
"""

import os
import glob
import time
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.features import rasterize
from shapely.geometry import box

# uint8 rasters => 256 possible values. Values are 0-100; 255 is nodata.
N_VALUES = 256
NODATA = 255
ALL_ZONE = "ALL"          # name of the global (whole-dataset) zone
UNZONED = "(unzoned)"     # pixels falling outside every polygon (zone id 0)

# Read this many rows at a time. 2048 rows x ~9000 cols x uint8 ~= 18 MB per
# chunk, so peak memory is tiny and independent of raster height.
CHUNK_ROWS = 2048


def prepare_zones(zones_file, zone_col, dst_crs):
    """Load a polygon file, dissolve by `zone_col`, reproject to `dst_crs`.

    Returns (shapes, id_to_name):
      shapes     : list of (geometry, zone_id) tuples for rasterize(), zone_id
                   an int >= 1 (0 is reserved for "outside every polygon").
      id_to_name : {zone_id: zone_name}, plus {0: UNZONED}.
    """
    gdf = gpd.read_file(zones_file)
    if zone_col not in gdf.columns:
        raise SystemExit(f"--zone_col '{zone_col}' not in {list(gdf.columns)}")

    # Drop null/empty zones, merge features sharing a name into one geometry.
    gdf = gdf[gdf[zone_col].notna() & gdf.geometry.notna()]
    gdf = gdf.dissolve(by=zone_col, as_index=False)[[zone_col, "geometry"]]
    gdf = gdf.to_crs(dst_crs)

    id_to_name = {0: UNZONED}
    shapes = []
    for zone_id, (_, row) in enumerate(gdf.iterrows(), start=1):
        id_to_name[zone_id] = str(row[zone_col])
        shapes.append((row.geometry, zone_id))
    print(f"Loaded {len(shapes)} zones from {os.path.basename(zones_file)} "
          f"(column '{zone_col}')", flush=True)
    return shapes, id_to_name


def _accumulate(block, counts, n_values=N_VALUES, zone_ids=None,
                zone_counts=None):
    """Add one raster window's value-counts into the running totals.

    Always updates the global `counts`. If `zone_ids` (a same-shape int label
    array) and `zone_counts` (a {zone_id: length-n_values array} dict) are
    given, also updates per-zone counts.
    """
    flat_vals = block.ravel()
    counts += np.bincount(flat_vals, minlength=n_values)

    if zone_ids is not None and zone_counts is not None:
        # One combined bincount over (zone_id * n_values + value) is far faster
        # than masking the array once per zone. The 2D result is (zone, value).
        flat_zones = zone_ids.ravel().astype(np.int32)
        max_zone = int(flat_zones.max())
        combined = flat_zones * n_values + flat_vals
        table = np.bincount(
            combined, minlength=(max_zone + 1) * n_values
        ).reshape(max_zone + 1, n_values)
        for zid in range(max_zone + 1):
            if table[zid].any():
                zone_counts.setdefault(zid, np.zeros(n_values, dtype=np.int64))
                zone_counts[zid] += table[zid]


def count_tif(tif_path, n_values=N_VALUES, chunk_rows=CHUNK_ROWS,
              zone_shapes=None, zone_dtype=np.uint16):
    """Count pixel values for one raster, optionally split by zone.

    Streams the raster in row-chunks so peak memory is bounded by one chunk.
    Returns (counts, zone_counts):
      counts      : length-n_values int64 array (global for this tif).
      zone_counts : {zone_id: length-n_values array}, or {} if no zones.
    """
    counts = np.zeros(n_values, dtype=np.int64)
    zone_counts = {}
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width

        tile_shapes = None
        if zone_shapes is not None:
            # Only rasterise polygons that actually touch this tile.
            tile_box = box(*src.bounds)
            tile_shapes = [(g, i) for (g, i) in zone_shapes
                           if g.intersects(tile_box)]

        for row in range(0, height, chunk_rows):
            nrows = min(chunk_rows, height - row)
            window = Window(0, row, width, nrows)
            block = src.read(1, window=window)

            zone_ids = None
            if zone_shapes is not None:
                if tile_shapes:
                    zone_ids = rasterize(
                        tile_shapes,
                        out_shape=(nrows, width),
                        transform=window_transform(window, src.transform),
                        fill=0,
                        dtype=zone_dtype,
                    )
                else:
                    # Tile touches no polygon: every pixel is unzoned (id 0),
                    # so it's still accounted for and zones sum to the total.
                    zone_ids = np.zeros((nrows, width), dtype=zone_dtype)
            _accumulate(block, counts, n_values, zone_ids, zone_counts)
            del block, zone_ids
    return counts, zone_counts


def counts_to_rows(zone_name, counts, nodata=NODATA, labels=None):
    """Turn a length-256 count vector for one zone into tidy row dicts."""
    valid_total = int(counts.sum() - counts[nodata])
    rows = []
    for value in range(len(counts)):
        if counts[value] == 0:
            continue
        is_nodata = value == nodata
        rows.append({
            "zone": zone_name,
            "value": value,
            "label": (labels or {}).get(value, ""),
            "count": int(counts[value]),
            "percent": (np.nan if is_nodata or valid_total == 0
                        else round(100 * counts[value] / valid_total, 6)),
            "is_nodata": is_nodata,
        })
    return rows


def run(tif_folder=None, suffix="", tif=None, output_csv=None,
        zones_file=None, zone_col=None, n_values=N_VALUES, nodata=NODATA,
        labels=None, limit=None):
    """Count pixel values across one tif or a folder, optionally by zone."""
    if tif:
        tif_files = [tif]
    else:
        tif_files = sorted(glob.glob(os.path.join(tif_folder, f"*{suffix}")))
    if limit is not None:
        tif_files = tif_files[:limit]
    if not tif_files:
        raise SystemExit("No TIF files found — check --tif/--tif_folder/--suffix")

    with rasterio.open(tif_files[0]) as src:
        dst_crs = src.crs

    zone_shapes, id_to_name = None, {}
    if zones_file:
        zone_shapes, id_to_name = prepare_zones(zones_file, zone_col, dst_crs)

    print(f"Counting pixel values across {len(tif_files)} tif(s)"
          + (" by zone" if zone_shapes else ""), flush=True)

    total = np.zeros(n_values, dtype=np.int64)
    zone_totals = {}  # zone_id -> length-n_values array
    t_start = time.time()
    for i, path in enumerate(tif_files, 1):
        t0 = time.time()
        counts, zone_counts = count_tif(path, n_values=n_values,
                                        zone_shapes=zone_shapes)
        total += counts
        for zid, zc in zone_counts.items():
            zone_totals.setdefault(zid, np.zeros(n_values, dtype=np.int64))
            zone_totals[zid] += zc
        print(f"  [{i}/{len(tif_files)}] {os.path.basename(path)} "
              f"({time.time() - t0:.1f}s)", flush=True)

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s ({elapsed / len(tif_files):.2f}s per tif)",
          flush=True)

    # Global "ALL" zone first, then each named zone.
    rows = counts_to_rows(ALL_ZONE, total, nodata=nodata, labels=labels)
    for zid in sorted(zone_totals):
        rows += counts_to_rows(id_to_name.get(zid, str(zid)),
                               zone_totals[zid], nodata=nodata, labels=labels)
    df = pd.DataFrame(rows)

    print(df[df.zone == ALL_ZONE].to_string(index=False), flush=True)

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}", flush=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count pixels of each value across a set of GeoTIFFs, "
                    "optionally within polygon zones.")
    parser.add_argument("--tif_folder", default=None,
                        help="Folder of tifs to count (globbed with --suffix).")
    parser.add_argument("--suffix", default=".tif",
                        help="Only files ending with this suffix are counted.")
    parser.add_argument("--tif", default=None,
                        help="Count a single tif instead of a folder.")
    parser.add_argument("--zones_file", default=None,
                        help="Polygon file (gpkg/shp) to split counts by.")
    parser.add_argument("--zone_col", default=None,
                        help="Attribute column in --zones_file to group by.")
    parser.add_argument("--output_csv", default=None,
                        help="Where to write the zone/value/count/percent CSV.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N tifs (for testing).")
    args = parser.parse_args()

    run(tif_folder=args.tif_folder, suffix=args.suffix, tif=args.tif,
        output_csv=args.output_csv, zones_file=args.zones_file,
        zone_col=args.zone_col, limit=args.limit)
