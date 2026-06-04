import glob
import os
import argparse

import geopandas as gpd
import pandas as pd


def merge_gpkg(base_dir, suffix='.gpkg', output_format='gpkg'):
    base_dir = base_dir.rstrip('/')
    glob_path = os.path.join(base_dir, f'*{suffix}')
    filenames = sorted(glob.glob(glob_path))

    if not filenames:
        raise FileNotFoundError(f"No files matching '*{suffix}' found in {base_dir}")

    print(f"Found {len(filenames)} files to merge")

    gdfs = []
    for i, f in enumerate(filenames):
        gdf = gpd.read_file(f)
        # gdf['source_file'] = os.path.basename(f)
        gdfs.append(gdf)
        if i % 10 == 0:
            print(f"  Read {i}/{len(filenames)}: {os.path.basename(f)}")

    merged = pd.concat(gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs=gdfs[0].crs)

    parent_dir = os.path.dirname(base_dir)
    parent_name = os.path.basename(parent_dir)
    base_name = os.path.basename(base_dir)

    if output_format == 'shp':
        outpath = os.path.join(parent_dir, f'merged_{parent_name}_{base_name}.shp')
        merged.to_file(outpath, driver='ESRI Shapefile')
    else:
        outpath = os.path.join(parent_dir, f'merged_{parent_name}_{base_name}.gpkg')
        merged.to_file(outpath, driver='GPKG')

    print(f"Saved {len(merged)} features to: {outpath}")
    return merged


def parse_arguments():
    parser = argparse.ArgumentParser(description='Merge all GeoPackage files in a folder into a single file.')
    parser.add_argument('base_dir', help='Directory containing the gpkg files to merge')
    parser.add_argument('--suffix', default='.gpkg', help='Filename suffix to match (default: .gpkg)')
    parser.add_argument('--output_format', default='gpkg', choices=['gpkg', 'shp'], help='Output format: gpkg or shp (default: gpkg)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    merge_gpkg(base_dir=args.base_dir, suffix=args.suffix, output_format=args.output_format)
