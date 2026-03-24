# Compare tree categories (from lidar) with model predictions across koppen regions
#
# Nick_indices: tree categories derived from lidar ground truth (categories 10-19 = trees)
# barra predictions: model's predicted percent tree cover (0-100)

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

np.random.seed(0)
random.seed(0)

# Directories
nick_indices_dir = '/scratch/xe2/cb8590/Nick_indices'
predictions_dir = '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2017/subfolders'
nick_outlines = '/g/data/xe2/cb8590/Nick_outlines'
koppen_australia = '/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned2.gpkg'


def get_prediction_tile(lat, lon, predictions_dir):
    """Get the prediction tile filename for a given lat/lon.
    
    Prediction tiles are named lat_XX_lon_YYY_merged_predicted.tif
    where XX is the even latitude and YYY is the even longitude,
    covering approximately 2 degrees in each direction.
    """
    lat_stub = int((abs(lat) // 2) * 2)
    lon_stub = int((lon // 2) * 2)
    filename = os.path.join(predictions_dir, f'lat_{lat_stub}_lon_{lon_stub}_merged_predicted.tif')
    return filename


def list_indices_tiles(indices_dir):
    """List all category tiles in the indices directory."""
    files = glob.glob(os.path.join(indices_dir, '*_linear_categories.tif'))
    tiles = []
    for f in files:
        basename = os.path.basename(f)
        stub = basename.replace('_linear_categories.tif', '')
        tiles.append({'stub': stub, 'indices_file': f})
    return tiles


def attach_koppen(matched_tiles):
    """Attach koppen climate classes to tiles using their centroids."""
    gdf_years = gpd.read_file(os.path.join(nick_outlines, 'tiff_footprints_years.gpkg'))
    gdf_years_3857 = gdf_years.to_crs("EPSG:3857")
    gdf_years_3857['stub'] = [f.split('.')[0] for f in gdf_years_3857['filename']]
    
    df_matched = pd.DataFrame(matched_tiles)
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


def evaluate_categories(tiles, predictions_dir, cover_threshold=50, limit=None, spacing=20):
    """Evaluate model prediction accuracy per tree category.
    
    For each Nick_indices tile:
    1. Read the category raster (lidar ground truth categories)
    2. Find which prediction tile contains it based on lat/lon
    3. Clip the prediction to the category tile extent
    4. Compare: categories 10-19 = tree (ground truth), prediction >= threshold = tree (model)
    """
    if limit:
        random.shuffle(tiles)
        tiles = tiles[:limit]
    
    print(f"Evaluating {len(tiles)} tiles")
    
    # Cache opened prediction tiles to avoid re-reading
    pred_cache = {}
    
    dfs = []
    skipped = 0
    for i, tile in enumerate(tiles):
        if i % 100 == 0:
            print(f"Working on {i}/{len(tiles)}")
        
        try:
            da_categories = rxr.open_rasterio(tile['indices_file']).isel(band=0).drop_vars('band')
            
            # Get tile centroid in EPSG:4326 to find the right prediction tile
            gs_bounds = gpd.GeoSeries([box(*da_categories.rio.bounds())], crs=da_categories.rio.crs)
            bounds_4326 = gs_bounds.to_crs('EPSG:4326').bounds.iloc[0]
            center_lat = (bounds_4326['miny'] + bounds_4326['maxy']) / 2
            center_lon = (bounds_4326['minx'] + bounds_4326['maxx']) / 2
            
            pred_file = get_prediction_tile(center_lat, center_lon, predictions_dir)
            if not os.path.exists(pred_file):
                skipped += 1
                continue
            
            # Open prediction tile (lazy with rioxarray)
            if pred_file not in pred_cache:
                pred_cache[pred_file] = rxr.open_rasterio(pred_file).isel(band=0).drop_vars('band')
            da_pred_full = pred_cache[pred_file]
            
            # Clip prediction to category tile bounds (in prediction CRS)
            bounds_pred_crs = gs_bounds.to_crs(da_pred_full.rio.crs).bounds.iloc[0]
            da_pred_clipped = da_pred_full.rio.clip_box(
                minx=bounds_pred_crs['minx'],
                miny=bounds_pred_crs['miny'],
                maxx=bounds_pred_crs['maxx'],
                maxy=bounds_pred_crs['maxy'],
            )
            
            # Reproject clipped prediction to match category tile
            da_pred_reprojected = da_pred_clipped.rio.reproject_match(
                da_categories, resampling=Resampling.nearest
            )
            
            # Binary: model predicts tree if percent cover >= threshold
            da_pred_trees = (da_pred_reprojected >= cover_threshold).astype(int)
            
            ds = xr.Dataset({
                'categories': da_categories,
                'pred_trees': da_pred_trees,
            })
            
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
        print(f"Skipped {skipped} tiles (missing prediction tile or error)")
    
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def compute_metrics(df_all):
    """Compute precision/recall/accuracy for each category.
    
    Ground truth: categories 10-19 = tree (from lidar)
    Prediction: pred_trees = 1 (model says tree)
    """
    y_true_trees = ((df_all['categories'] >= 10) & (df_all['categories'] < 20)).astype(int).values
    y_pred = df_all['pred_trees'].values
    
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
        
        # What fraction of pixels in this category does the model predict as tree?
        pred_in_cat = y_pred[mask]
        model_tree_fraction = pred_in_cat.mean()
        
        rows.append({
            'category': cat,
            'label': label,
            'is_tree_category': is_tree_category,
            'n_pixels': int(n_pixels),
            'fraction_model_predicts_tree': round(float(model_tree_fraction), 3),
        })
    
    df_categories = pd.DataFrame(rows)
    
    # Overall binary metrics
    overall = {
        'precision': round(precision_score(y_true_trees, y_pred, zero_division=0), 3),
        'recall': round(recall_score(y_true_trees, y_pred, zero_division=0), 3),
        'accuracy': round(accuracy_score(y_true_trees, y_pred), 3),
    }
    
    # Per-koppen metrics
    koppen_rows = []
    if 'koppen_class' in df_all.columns:
        for koppen_class in sorted(df_all['koppen_class'].dropna().unique()):
            df_k = df_all[df_all['koppen_class'] == koppen_class]
            y_true_k = ((df_k['categories'] >= 10) & (df_k['categories'] < 20)).astype(int).values
            y_pred_k = df_k['pred_trees'].values
            
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
    parser = argparse.ArgumentParser(description="Evaluate model prediction accuracy per tree category")
    parser.add_argument("--limit", type=int, default=None, help="Number of tiles to evaluate")
    parser.add_argument("--spacing", type=int, default=20, help="Jittered grid spacing")
    parser.add_argument("--cover-threshold", type=int, default=50, help="Percent cover threshold for model predictions (default: 50)")
    parser.add_argument("--predictions-dir", default=predictions_dir, help="Directory containing prediction tiles")
    parser.add_argument("--with-koppen", action="store_true", help="Attach koppen climate classes (slower)")
    args = parser.parse_args()
    
    print("Listing indices tiles...")
    tiles = list_indices_tiles(nick_indices_dir)
    print(f"Found {len(tiles)} indices tiles")
    
    if args.with_koppen:
        print("Attaching koppen classes...")
        gdf = attach_koppen(tiles)
        tiles = gdf.to_dict('records')
    
    print(f"Evaluating categories (cover_threshold={args.cover_threshold})...")
    df_all = evaluate_categories(tiles, args.predictions_dir, 
                                  cover_threshold=args.cover_threshold,
                                  limit=args.limit, spacing=args.spacing)
    
    print(f"\nTotal sampled pixels: {len(df_all)}")
    
    df_categories, overall, df_koppen = compute_metrics(df_all)
    
    print("\n=== Per-category breakdown ===")
    print("(fraction_model_predicts_tree: should be ~1.0 for tree categories, ~0.0 for non-tree)")
    print(df_categories.to_string(index=False))
    
    print(f"\n=== Overall binary metrics (categories 10-19 = tree ground truth) ===")
    print(f"  Precision: {overall['precision']}")
    print(f"  Recall:    {overall['recall']}")
    print(f"  Accuracy:  {overall['accuracy']}")
    
    if len(df_koppen) > 0:
        print("\n=== Per-koppen metrics ===")
        print(df_koppen.to_string(index=False))


if __name__ == '__main__':
    main()
