"""
Create a filtered copy (via symlinks) of the expanded BARRA trees tile directory,
keeping only tiles that spatially intersect the grazing/no-BWH bounding boxes.

Tile filenames encode the centroid: e.g. "11_13-131_98_y2020_..." → lat=-11.13, lon=131.98.
Each tile is ~0.044° across. We build a small bounding box per tile and query a spatial
index of the mask polygons to decide inclusion.

Output directory mirrors the lat_XX_lon_XXX subfolder structure of the source.
"""

import os
import re
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

SRC = Path("/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/expanded")
DST = Path("/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/expanded_grazing_no_bwh")
MASK = Path("/g/data/xe2/cb8590/Nick_outlines/barra_bboxs_grazing_no_bwh.gpkg")

# Approximate half-width of each tile in degrees (tiles are ~0.044°; 0.025 adds a safe margin)
TILE_HALF = 0.025

_TILE_RE = re.compile(r"^(\d+)_(\d+)-(\d+)_(\d+)_")


def tile_bbox(filename: str):
    """Return a Shapely box for a tile, or None if the filename doesn't match."""
    m = _TILE_RE.match(filename)
    if not m:
        return None
    lat = -(float(f"{m.group(1)}.{m.group(2)}"))
    lon = float(f"{m.group(3)}.{m.group(4)}")
    return box(lon - TILE_HALF, lat - TILE_HALF, lon + TILE_HALF, lat + TILE_HALF)


def main():
    print(f"Loading mask: {MASK}")
    gdf = gpd.read_file(MASK).to_crs("EPSG:4326")
    sindex = gdf.sindex
    print(f"  {len(gdf):,} features loaded")

    n_linked = 0
    n_skipped = 0

    for tile_dir in sorted(SRC.iterdir()):
        if not tile_dir.is_dir():
            continue

        tifs = sorted(f for f in tile_dir.iterdir() if f.suffix == ".tif")
        if not tifs:
            continue

        for tif in tifs:
            bbox = tile_bbox(tif.name)
            if bbox is None:
                print(f"  WARNING: could not parse {tif.name}")
                n_skipped += 1
                continue

            if not list(sindex.intersection(bbox.bounds)):
                n_skipped += 1
                continue

            dst_dir = DST / tile_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_link = dst_dir / tif.name
            if not dst_link.exists():
                os.symlink(tif.resolve(), dst_link)
            n_linked += 1

    print(f"\nDone. Symlinked: {n_linked}, skipped: {n_skipped}")
    print(f"Output: {DST}")


if __name__ == "__main__":
    main()
