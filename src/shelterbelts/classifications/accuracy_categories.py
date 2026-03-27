# Compare Nick's tree categories against multiple prediction sources
#
# Ground truth: Nick_indices (categories 10-19 = trees)
# Prediction sources:
#   1. barra model predictions (percent tree cover, tree if >= 50%)
#   2. Nick_worldcover_reprojected (ESA WorldCover, tree if class == 10)
#   3. Nick_GCH (Global Canopy Height, tree if height >= 1m)

import os
import glob
import argparse
import random

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import box
from sklearn.metrics import accuracy_score, precision_score, recall_score

from shelterbelts.classifications.merge_inputs_outputs import jittered_grid
from shelterbelts.indices.all_indices import GEE_legend

np.random.seed(1)
random.seed(1)

# Directories
nick_indices_dir = '/scratch/xe2/cb8590/Nick_indices'
nick_worldcover_dir = '/scratch/xe2/cb8590/Nick_worldcover_reprojected'
nick_gch_dir = '/scratch/xe2/cb8590/Nick_GCH'
predictions_dir = '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2017/subfolders'
nick_outlines = '/g/data/xe2/cb8590/Nick_outlines'
koppen_australia = '/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned2.gpkg'

SOURCES = ['barra', 'worldcover', 'gch']


def get_prediction_tile(lat, lon, predictions_dir):
    """Get the barra prediction tile filename for a given lat/lon."""
    lat_stub = int((abs(lat) // 2) * 2)
    lon_stub = int((lon // 2) * 2)
    return os.path.join(predictions_dir, f'lat_{lat_stub}_lon_{lon_stub}_merged_predicted.tif')


def list_common_tiles():
    """Find tiles present in all three Nick directories (indices, worldcover, GCH)."""
    indices_files = glob.glob(os.path.join(nick_indices_dir, '*_linear_categories.tif'))
    worldcover_files = glob.glob(os.path.join(nick_worldcover_dir, '*_worldcover.tif'))
    gch_files = glob.glob(os.path.join(nick_gch_dir, '*_canopy_height.tif'))

    indices_stubs = {os.path.basename(f).replace('_linear_categories.tif', ''): f for f in indices_files}
    worldcover_stubs = {os.path.basename(f).replace('_worldcover.tif', ''): f for f in worldcover_files}
    gch_stubs = {os.path.basename(f).replace('_canopy_height.tif', ''): f for f in gch_files}

    common = set(indices_stubs) & set(worldcover_stubs) & set(gch_stubs)

    tiles = []
    for stub in sorted(common):
        tiles.append({
            'stub': stub,
            'indices_file': indices_stubs[stub],
            'worldcover_file': worldcover_stubs[stub],
            'gch_file': gch_stubs[stub],
        })

    return tiles


def attach_koppen(tiles):
    """Attach koppen climate classes to tiles using their centroids."""
    gdf_years = gpd.read_file(os.path.join(nick_outlines, 'tiff_footprints_years.gpkg'))
    gdf_years_3857 = gdf_years.to_crs("EPSG:3857")
    gdf_years_3857['stub'] = [f.split('.')[0] for f in gdf_years_3857['filename']]

    df_matched = pd.DataFrame(tiles)
    gdf_matched = df_matched.merge(gdf_years_3857[['stub', 'geometry']], on='stub', how='inner')
    gdf_matched = gpd.GeoDataFrame(gdf_matched, geometry='geometry', crs="EPSG:3857")

    gdf_koppen = gpd.read_file(koppen_australia).to_crs("EPSG:3857")
    gdf_joined = gdf_matched.sjoin_nearest(
        gdf_koppen[['geometry', 'Name']],
        how="left",
        distance_col="distance_to_koppen"
    )
    gdf_joined = gdf_joined.rename(columns={'Name': 'koppen_class'})

    return gdf_joined


def evaluate_categories(tiles, predictions_dir, limit=None, spacing=20):
    """Evaluate prediction accuracy per tree category for all sources.

    For each tile:
    1. Read categories (lidar ground truth)
    2. Get binary tree predictions from each source:
       - barra: clip from 2x2 degree tile, reproject, >= 50
       - worldcover: reproject to match indices, == 10
       - gch: reproject to match indices, >= 2m
    """
    if limit:
        random.shuffle(tiles)
        tiles = tiles[:limit]

    print(f"Evaluating {len(tiles)} tiles")

    pred_cache = {}

    dfs = []
    skipped = 0
    for i, tile in enumerate(tiles):
        if i % 100 == 0:
            print(f"Working on {i}/{len(tiles)}")

        try:
            da_categories = rxr.open_rasterio(tile['indices_file']).isel(band=0).drop_vars('band')

            gs_bounds = gpd.GeoSeries([box(*da_categories.rio.bounds())], crs=da_categories.rio.crs)
            bounds_4326 = gs_bounds.to_crs('EPSG:4326').bounds.iloc[0]
            center_lat = (bounds_4326['miny'] + bounds_4326['maxy']) / 2
            center_lon = (bounds_4326['minx'] + bounds_4326['maxx']) / 2

            ds_dict = {'categories': da_categories}

            # --- Barra ---
            pred_file = get_prediction_tile(center_lat, center_lon, predictions_dir)
            if os.path.exists(pred_file):
                if pred_file not in pred_cache:
                    pred_cache[pred_file] = rxr.open_rasterio(pred_file).isel(band=0).drop_vars('band')
                da_pred_full = pred_cache[pred_file]
                bounds_pred_crs = gs_bounds.to_crs(da_pred_full.rio.crs).bounds.iloc[0]
                da_pred_clipped = da_pred_full.rio.clip_box(
                    minx=bounds_pred_crs['minx'], miny=bounds_pred_crs['miny'],
                    maxx=bounds_pred_crs['maxx'], maxy=bounds_pred_crs['maxy'],
                )
                da_barra = da_pred_clipped.rio.reproject_match(
                    da_categories, resampling=Resampling.nearest
                )
                ds_dict['barra'] = (da_barra >= 50).astype(int)

            # --- Worldcover ---
            da_wc = rxr.open_rasterio(tile['worldcover_file']).isel(band=0).drop_vars('band')
            da_wc_reproj = da_wc.rio.reproject_match(da_categories, resampling=Resampling.nearest)
            ds_dict['worldcover'] = (da_wc_reproj == 10).astype(int)

            # --- GCH ---
            da_gch = rxr.open_rasterio(tile['gch_file']).isel(band=0).drop_vars('band')
            da_gch_reproj = da_gch.rio.reproject_match(da_categories, resampling=Resampling.nearest)
            ds_dict['gch'] = (da_gch_reproj >= 1).astype(int)

            ds = xr.Dataset(ds_dict)
            df = jittered_grid(ds, spacing=spacing)
            df['stub'] = tile['stub']
            if 'koppen_class' in tile:
                df['koppen_class'] = tile['koppen_class']
            dfs.append(df)

        except Exception as e:
            print(f"  Error on {tile['stub']}: {e}")
            skipped += 1
            continue

    if skipped:
        print(f"Skipped {skipped} tiles")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def compute_metrics(df_all, source):
    """Compute precision/recall/accuracy for each category for a given source."""
    y_true_trees = ((df_all['categories'] >= 10) & (df_all['categories'] < 20)).astype(int).values
    y_pred = df_all[source].values

    categories = sorted(df_all['categories'].unique())

    rows = []
    for cat in categories:
        cat = int(cat)
        label = GEE_legend.get(cat, f'Unknown ({cat})')
        is_tree_category = 10 <= cat < 20

        mask = df_all['categories'] == cat
        n_pixels = mask.sum()
        if n_pixels == 0:
            continue

        pred_in_cat = y_pred[mask]
        fraction_predicts_tree = pred_in_cat.mean()

        rows.append({
            'category': cat,
            'label': label,
            'is_tree_category': is_tree_category,
            'n_pixels': int(n_pixels),
            'fraction_predicts_tree': round(float(fraction_predicts_tree), 3),
        })

    df_categories = pd.DataFrame(rows)

    overall = {
        'precision': round(precision_score(y_true_trees, y_pred, zero_division=0), 3),
        'recall': round(recall_score(y_true_trees, y_pred, zero_division=0), 3),
        'accuracy': round(accuracy_score(y_true_trees, y_pred), 3),
    }

    koppen_rows = []
    if 'koppen_class' in df_all.columns:
        for koppen_class in sorted(df_all['koppen_class'].dropna().unique()):
            df_k = df_all[df_all['koppen_class'] == koppen_class]
            y_true_k = ((df_k['categories'] >= 10) & (df_k['categories'] < 20)).astype(int).values
            y_pred_k = df_k[source].values

            koppen_rows.append({
                'koppen_class': koppen_class,
                'n_pixels': len(df_k),
                'precision': round(precision_score(y_true_k, y_pred_k, zero_division=0), 3),
                'recall': round(recall_score(y_true_k, y_pred_k, zero_division=0), 3),
                'accuracy': round(accuracy_score(y_true_k, y_pred_k), 3),
            })

    df_koppen = pd.DataFrame(koppen_rows)
    return df_categories, overall, df_koppen


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy per tree category for multiple sources")
    parser.add_argument("--limit", type=int, default=None, help="Number of tiles to evaluate")
    parser.add_argument("--spacing", type=int, default=20, help="Jittered grid spacing")
    parser.add_argument("--predictions-dir", default=predictions_dir, help="Barra predictions directory")
    args = parser.parse_args()

    print("Finding common tiles across indices, worldcover and GCH...")
    tiles = list_common_tiles()
    print(f"Found {len(tiles)} common tiles")

    print("Attaching koppen classes...")
    gdf = attach_koppen(tiles)
    tiles = gdf.to_dict('records')

    print("Evaluating categories...")
    df_all = evaluate_categories(tiles, args.predictions_dir,
                                  limit=args.limit, spacing=args.spacing)

    print(f"\nTotal sampled pixels: {len(df_all)}")

    for source in SOURCES:
        if source not in df_all.columns:
            print(f"\n=== {source.upper()} ===")
            print("  (no data available)")
            continue

        df_source = df_all.dropna(subset=[source])
        df_categories, overall, df_koppen = compute_metrics(df_source, source)

        print(f"\n{'='*60}")
        print(f"=== {source.upper()} ===")
        print(f"{'='*60}")

        print("\nPer-category breakdown:")
        print("(fraction_predicts_tree: should be ~1.0 for tree categories, ~0.0 for non-tree)")
        print(df_categories.to_string(index=False))

        print(f"\nOverall binary metrics (categories 10-19 = tree ground truth):")
        print(f"  Precision: {overall['precision']}")
        print(f"  Recall:    {overall['recall']}")
        print(f"  Accuracy:  {overall['accuracy']}")

        if len(df_koppen) > 0:
            print(f"\nPer-koppen metrics:")
            print(df_koppen.to_string(index=False))


if __name__ == '__main__':
    main()
