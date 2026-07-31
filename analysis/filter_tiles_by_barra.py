"""
Crop tiles_30000_New_South_Wales.gpkg to only tiles that overlap with at least
one tile in barra_bboxs_grazing_no_bwh.gpkg.
"""

import time
import geopandas as gpd
from pathlib import Path

TILES_PATH = Path("outdir/tiles_30000_New_South_Wales.gpkg")
BARRA_PATH = Path("outdir/barra_bboxs_grazing_no_bwh.gpkg")
OUTPUT_PATH = Path("outdir/tiles_30000_New_South_Wales_barra_overlap.gpkg")
SAMPLE_SIZE = 50  # number of barra tiles to use for the sample run


def filter_tiles(sample=False):
    print("Loading tiles...")
    t0 = time.time()
    tiles = gpd.read_file(TILES_PATH)
    print(f"  {len(tiles)} NSW tiles loaded ({time.time() - t0:.1f}s)")

    print("Loading barra bboxes...")
    t1 = time.time()
    barra = gpd.read_file(BARRA_PATH)
    print(f"  {len(barra)} barra tiles loaded ({time.time() - t1:.1f}s)")

    if sample:
        barra = barra.iloc[:SAMPLE_SIZE].copy()
        print(f"  Sample mode: using first {SAMPLE_SIZE} barra tiles")

    # Reproject tiles to barra CRS (fewer rows to reproject)
    print(f"Reprojecting NSW tiles from {tiles.crs} to {barra.crs}...")
    t2 = time.time()
    tiles_reproj = tiles.to_crs(barra.crs)
    print(f"  Done ({time.time() - t2:.1f}s)")

    # Dissolve barra into a single unary union for fast containment check
    print("Dissolving barra tiles into union geometry...")
    t3 = time.time()
    barra_union = barra.unary_union
    print(f"  Done ({time.time() - t3:.1f}s)")

    # Find NSW tiles that intersect the barra union
    print("Finding overlapping NSW tiles...")
    t4 = time.time()
    overlaps = tiles_reproj.geometry.intersects(barra_union)
    matching_tiles = tiles[overlaps].copy()
    print(f"  {overlaps.sum()} / {len(tiles)} tiles overlap ({time.time() - t4:.1f}s)")

    if not sample:
        print(f"Saving to {OUTPUT_PATH}...")
        matching_tiles.to_file(OUTPUT_PATH, driver="GPKG")
        print(f"  Saved {len(matching_tiles)} tiles")

    total = time.time() - t0
    print(f"Total time: {total:.1f}s")
    return matching_tiles, total


if __name__ == "__main__":
    import sys

    full_run = "--full" in sys.argv

    if not full_run:
        print("=== SAMPLE RUN (first 50 barra tiles) ===")
        result, elapsed = filter_tiles(sample=True)
        print(f"\nSample completed in {elapsed:.1f}s.")
        print("Run with --full to process all tiles and save output.")
    else:
        print("=== FULL RUN ===")
        result, elapsed = filter_tiles(sample=False)
