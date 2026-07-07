"""
Delete small per-tile prediction tifs that fall outside the grazing/no-BWH agricultural mask, 
freeing up scratch space + inodes ahead of running the shelterbelts pipeline 
(which only needs tiles inside agricultural regions).
"""

import argparse
import re
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

MASK = Path("/g/data/xe2/cb8590/Nick_outlines/barra_bboxs_grazing_no_bwh.gpkg")

# Approximate half-width of each tile in degrees (tiles are ~0.044 deg; 0.025 adds a safe margin)
TILE_HALF = 0.025

_TILE_DIR_RE = re.compile(r"^lat_\d+_lon_\d+$")
_TILE_FILE_RE = re.compile(r"^(\d+)_(\d+)-(\d+)_(\d+)_")


def tile_bbox(filename: str):
    """Return a Shapely box for a tile, or None if the filename doesn't match."""
    m = _TILE_FILE_RE.match(filename)
    if not m:
        return None
    lat = -(float(f"{m.group(1)}.{m.group(2)}"))
    lon = float(f"{m.group(3)}.{m.group(4)}")
    return box(lon - TILE_HALF, lat - TILE_HALF, lon + TILE_HALF, lat + TILE_HALF)


def find_tile_dirs(src: Path):
    return sorted(d for d in src.iterdir() if d.is_dir() and _TILE_DIR_RE.match(d.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Directory containing lat_XX_lon_XXX subfolders")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't delete")
    args = parser.parse_args()

    print(f"Loading mask: {MASK}")
    gdf = gpd.read_file(MASK).to_crs("EPSG:4326")
    sindex = gdf.sindex
    print(f"  {len(gdf):,} features loaded")

    tile_dirs = find_tile_dirs(args.src)
    print(f"Found {len(tile_dirs)} lat_XX_lon_XXX subdirectories under {args.src}")

    before_count = 0
    before_size = 0
    to_delete = []
    n_unparsed = 0

    for tile_dir in tile_dirs:
        for f in tile_dir.iterdir():
            if f.suffix != ".tif":
                continue
            size = f.stat().st_size
            before_count += 1
            before_size += size

            bbox = tile_bbox(f.name)
            if bbox is None:
                n_unparsed += 1
                continue
            if not list(sindex.intersection(bbox.bounds)):
                to_delete.append((f, size))

    print(f"\nBefore: {before_count:,} files, {before_size / 1e9:.2f} GB")
    if n_unparsed:
        print(f"  WARNING: {n_unparsed:,} files didn't match the tile filename pattern (left untouched)")

    del_size = sum(size for _, size in to_delete)
    print(f"Outside agricultural regions: {len(to_delete):,} files, {del_size / 1e9:.2f} GB")

    if args.dry_run:
        print("Dry run - not deleting anything")
        return

    for f, _ in to_delete:
        f.unlink()

    after_count = before_count - len(to_delete)
    after_size = before_size - del_size
    print(f"\nAfter: {after_count:,} files, {after_size / 1e9:.2f} GB")
    print(f"Freed: {len(to_delete):,} files, {del_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
