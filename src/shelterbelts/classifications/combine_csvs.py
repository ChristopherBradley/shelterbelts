import os
import glob
import argparse

import geopandas as gpd
import pandas as pd
from pyproj import Transformer

from shelterbelts.utils.filepaths import koppen_australia


def combine_csvs(csv_folder, outdir=None, crs_csv=None, limit=None):
    """
    Merge per-tile training CSVs into a single feather file and attach koppen classes.

    Parameters
    ----------
    csv_folder : str
        Folder containing ``*.csv`` training files produced by
        :func:`shelterbelts.classifications.merge_inputs_outputs.merge_inputs_outputs`.
    outdir : str, optional
        Output directory for saving results. Defaults to csv_folder.
    crs_csv : str, optional
        CSV with tif and crs columns that assigns a source CRS to each tile.
    limit : int, optional
        Read only the first CSVs. By default it reads all.

    Returns
    -------
    pandas.DataFrame
        The concatenated, reprojected, Koppen-tagged training set.

    """
    if outdir is None:
        outdir = csv_folder

    files = sorted(glob.glob(os.path.join(csv_folder, 'g*.csv')))
    limit_stub = ""
    if limit:
        files = files[:limit]
        limit_stub = f"_{limit}"

    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    filename = os.path.join(outdir, f'df_all{limit_stub}.feather')
    df_all.to_feather(filename)
    print(f"Saved: {filename}")

    if crs_csv is not None:
        df_crs = pd.read_csv(crs_csv)
        df_crs['tile_id'] = [row['tif'].split('.')[0] for _, row in df_crs.iterrows()]
        df_all = df_all.merge(df_crs[['tile_id', 'crs']])

        # Reproject each CRS group to EPSG:4326 so all tiles share a frame.
        dfs = []
        for crs, group in df_all.groupby('crs'):
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            x2, y2 = transformer.transform(group['x'].values, group['y'].values)
            group = group.copy()
            group['x'], group['y'], group['crs'] = x2, y2, 'EPSG:4326'
            dfs.append(group)
        df_all = pd.concat(dfs, ignore_index=True)
        filename = os.path.join(outdir, f'df_all_4326{limit_stub}.feather')
        df_all.to_feather(filename)
        print(f"Saved: {filename}")

    gdf_koppen = gpd.read_file(koppen_australia)
    gdf_points = gpd.GeoDataFrame(
        df_all,
        geometry=gpd.points_from_xy(df_all['x'], df_all['y']),
        crs="EPSG:4326",
    )
    gdf_joined = gpd.sjoin(gdf_points, gdf_koppen, how='left', predicate='within')
    df_all['Koppen'] = gdf_joined['Name'].values
    filename = os.path.join(outdir, 'df_all_koppen.feather')
    df_all.to_feather(filename)
    print(f"Saved: {filename}")

    for koppen_class in df_all['Koppen'].unique():
        df_class = df_all[df_all['Koppen'] == koppen_class]
        filename = os.path.join(outdir, f'df_4326_{koppen_class}{limit_stub}.feather')
        df_class.to_feather(filename)
        print(f"Saved: {filename}")

    return df_all


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing g*.csv training files")
    parser.add_argument('--outdir', default=None, help='Output directory (default: same as folder)')
    parser.add_argument('--crs_csv', default=None, help='CSV mapping tile_id to source CRS for reprojecting to EPSG:4326')
    parser.add_argument('--limit', type=int, default=None, help='Read only the first N csvs (default: all)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    combine_csvs(args.folder, outdir=args.outdir, crs_csv=args.crs_csv, limit=args.limit)
