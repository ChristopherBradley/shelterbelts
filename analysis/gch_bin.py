"""Phase 1: bin GCH v2 30 km tiles to global-lattice 10 m percent + height rasters (chunked for PBS).

Only tiles that actually intersect an ag bbox are binned. Pass --make_list once to precompute the
needed-tile list, then bin jobs process a [--start:--end] slice of it.

    python gch_bin.py --make_list                       # writes needed_gch_tiles.txt
    python gch_bin.py --start 0 --end 100               # bin tiles 0..99 of that list
"""
import os
import glob
import argparse

import geopandas as gpd

from gch_prep import bin_gch_tiles

GCH_DIR = '/scratch/xe2/cb8590/Global_Canopy_Height_v2'
BINNED_DIR = '/scratch/xe2/cb8590/gch_v2_ag_indices/binned'
FOOTPRINTS = '/g/data/xe2/cb8590/Outlines/global_canopy_height_v2_footprints.gpkg'
AG_BBOXS = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_ag.gpkg'
TILE_LIST = '/scratch/xe2/cb8590/gch_v2_ag_indices/needed_gch_tiles.txt'


def make_list():
    fp = gpd.read_file(FOOTPRINTS)                                   # EPSG:3857
    ag = gpd.read_file(AG_BBOXS).set_crs(4326, allow_override=True).to_crs(3857)
    sidx = ag.sindex
    needed = [fn for fn, geom in zip(fp['filename'], fp.geometry)
              if len(sidx.query(geom, predicate='intersects')) > 0]
    os.makedirs(os.path.dirname(TILE_LIST), exist_ok=True)
    with open(TILE_LIST, 'w') as f:
        f.write('\n'.join(sorted(needed)) + '\n')
    print(f"{len(needed)} of {len(fp)} GCH tiles intersect ag -> {TILE_LIST}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--make_list', action='store_true')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=None)
    p.add_argument('--gch_dir', default=GCH_DIR)
    p.add_argument('--binned_dir', default=BINNED_DIR)
    p.add_argument('--list', default=TILE_LIST, help='Tile-list file to slice (default: needed_gch_tiles.txt)')
    args = p.parse_args()

    if args.make_list:
        make_list()
        return

    with open(args.list) as f:
        tiles = [ln.strip() for ln in f if ln.strip()]
    chunk = tiles[args.start:args.end]
    paths = [os.path.join(args.gch_dir, t) for t in chunk]
    print(f"Binning {len(paths)} tiles [{args.start}:{args.end}] -> {args.binned_dir}", flush=True)
    n = bin_gch_tiles(paths, args.binned_dir)
    print(f"Binned {n} new tiles (skipped {len(paths)-n} already done)", flush=True)


if __name__ == '__main__':
    main()
