"""Phase 2a: cut one ~200 km region's ag 4 km tiles (expanded) out of the binned 10 m GCH mosaic.

    python gch_cut.py --region lat_34_lon_148 --out_dir <region_input_dir> [--overlap_px 50]
"""
import argparse
import geopandas as gpd

from gch_prep import cut_region

BINNED_DIR = '/scratch/xe2/cb8590/gch_v2_ag_indices/binned'
FOOTPRINTS = '/g/data/xe2/cb8590/Outlines/global_canopy_height_v2_footprints.gpkg'
AG_BBOXS = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_ag.gpkg'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--region', required=True)
    p.add_argument('--out_dir', required=True, help='Region input dir; percent/ + height/ written here')
    p.add_argument('--binned_dir', default=BINNED_DIR)
    p.add_argument('--footprints', default=FOOTPRINTS)
    p.add_argument('--ag_bboxs', default=AG_BBOXS)
    p.add_argument('--overlap_px', type=int, default=50)
    p.add_argument('--no_height', action='store_true', default=False,
                   help="Only write percent/ (for runs that don't use height-aware shelter)")
    args = p.parse_args()

    bboxs = gpd.read_file(args.ag_bboxs).set_crs(4326, allow_override=True)
    fp = gpd.read_file(args.footprints)
    n, total = cut_region(args.region, bboxs, args.binned_dir, fp, args.out_dir,
                          overlap_px=args.overlap_px, save_height=not args.no_height)
    print(f"[{args.region}] cut {n}/{total} tiles -> {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
