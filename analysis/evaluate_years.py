# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluating barra_trees predictions across years (2017, 2020, 2024)
#
# For each tile in gdf_nodesert, crops the merged_predicted.tif for each year
# to the tile's bounding box, samples a jittered grid of pixels, and computes
# precision / recall / accuracy against Nick's ground-truth labels,
# stratified by Köppen zone.

# %%
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box
from sklearn.metrics import accuracy_score, precision_score, recall_score

from shelterbelts.classifications.merge_inputs_outputs import jittered_grid
from shelterbelts.utils.filepaths import (
    koppen_australia,
    nick_aus_treecover_10m,
    nick_outlines,
)

np.random.seed(0)

PRED_DIRS = {
    2017: '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2017/subfolders',
    2020: '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/subfolders',
    2024: '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2024/subfolders',
}
OUTPUT_CSV = f'{nick_outlines}/df_evaluation_years.csv'
THRESHOLD = 50  # predictions are 0–100 probability; >= 50 → tree


# %%
def build_tif_index(pred_dir):
    """GDF of merged_predicted.tif footprints in EPSG:3857 for spatial lookup."""
    files = [f for f in os.listdir(pred_dir) if f.endswith('_merged_predicted.tif')]
    records = []
    for f in files:
        path = os.path.join(pred_dir, f)
        with rasterio.open(path) as src:
            b = src.bounds
        records.append({'path': path, 'geometry': box(b.left, b.bottom, b.right, b.top)})
    return gpd.GeoDataFrame(records, crs='EPSG:3857')


def windowed_da(tif_path, bbox_3857):
    """Read a window from a merged tif (EPSG:3857) and return an xr.DataArray."""
    with rasterio.open(tif_path) as src:
        window = from_bounds(*bbox_3857, src.transform)
        data = src.read(1, window=window)
        transform = src.window_transform(window)
    h, w = data.shape
    if h == 0 or w == 0:
        return None
    x = transform.c + (np.arange(w) + 0.5) * transform.a
    y = transform.f + (np.arange(h) + 0.5) * transform.e
    da = xr.DataArray(data, dims=('y', 'x'), coords={'y': y, 'x': x})
    return da.rio.write_crs('EPSG:3857')


# %%
def build_sample_df():
    """Process all nodesert tiles and return sampled pixel comparison DataFrame."""
    gdf = gpd.read_file(f'{nick_outlines}/footprints_large_recent_nodesert.gpkg')
    gdf['stub'] = [f.split('.')[0] for f in gdf['filename']]
    gdf_koppen = gpd.read_file(koppen_australia).to_crs(gdf.crs)
    gdf = (
        gdf.sjoin_nearest(gdf_koppen[['geometry', 'Name']], how='left', distance_col='_d')
           .drop(columns=['index_right', '_d'], errors='ignore')
    )
    gdf_3857 = gdf.to_crs('EPSG:3857')

    # Build spatial lookup: tile stub → merged tif path, per year
    print("Building tif indexes and spatial joins...")
    joins = {}
    for yr, pred_dir in PRED_DIRS.items():
        tidx = build_tif_index(pred_dir)
        joined = gpd.sjoin(
            gdf_3857[['stub', 'Name', 'geometry']], tidx,
            how='left', predicate='intersects'
        )
        joins[yr] = (
            joined.drop_duplicates(subset='stub', keep='first')
                  .set_index('stub')['path']
        )
        n_matched = joined['path'].notna().sum()
        print(f"  {yr}: {n_matched}/{len(gdf)} tiles matched a merged tif")

    # Process each tile
    rows = []
    n = len(gdf_3857)
    for i, (_, tile_row) in enumerate(gdf_3857.iterrows()):
        stub = tile_row['stub']
        if i % 250 == 0:
            print(f"  tile {i}/{n}: {stub}")

        label_path = f'{nick_aus_treecover_10m}/{stub}.tiff'
        if not os.path.exists(label_path):
            continue

        # Windowed read for each year
        pred_das = {}
        for yr in PRED_DIRS:
            tif_path = joins[yr].get(stub)
            if not isinstance(tif_path, str):
                continue
            da = windowed_da(tif_path, tile_row.geometry.bounds)
            if da is not None:
                pred_das[yr] = (da >= THRESHOLD).astype(int)

        if len(pred_das) < len(PRED_DIRS):
            continue

        ds_label = rxr.open_rasterio(label_path).isel(band=0).drop_vars('band').astype(int)
        ref = pred_das[2017]
        ds = xr.Dataset({'nick_trees': ds_label.rio.reproject_match(ref, resampling=Resampling.nearest)})
        for yr, da in pred_das.items():
            ds[f'pred_{yr}'] = da.rio.reproject_match(ref, resampling=Resampling.nearest)

        df = jittered_grid(ds, spacing=20)
        df['stub'] = stub
        df['koppen_class'] = tile_row['Name']
        rows.append(df)

    df_all = pd.concat(rows, ignore_index=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_all)} samples to {OUTPUT_CSV}")
    return df_all


# %%
# %%time
if os.path.exists(OUTPUT_CSV):
    print(f"Loading existing results from {OUTPUT_CSV}")
    df_all = pd.read_csv(OUTPUT_CSV)
else:
    print("Computing samples (~20 min)...")
    df_all = build_sample_df()

# %% [markdown]
# ## Precision / recall / accuracy per Köppen zone per year

# %%
koppen_order = ['Aw', 'BSh', 'BSk', 'CFa', 'Cfb']
models = [f'pred_{yr}' for yr in PRED_DIRS]

rows = []
for koppen_class in koppen_order:
    df = df_all[df_all['koppen_class'] == koppen_class].copy()
    # Drop edge-pixel artifacts (reproject_match nodata sentinel or NaN)
    valid_cols = ['nick_trees'] + [m for m in models if m in df.columns]
    for col in valid_cols:
        df = df[df[col].isin([0, 1])]
    if df.empty:
        continue
    row = {'koppen_class': koppen_class}
    for model in models:
        if model not in df.columns:
            continue
        y_true, y_pred = df['nick_trees'].astype(int), df[model].astype(int)
        row[(model, 'precision')] = round(precision_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'recall')]    = round(recall_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'accuracy')]  = round(accuracy_score(y_true, y_pred), 2)
    rows.append(row)

results_df = pd.DataFrame(rows).set_index('koppen_class')
results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)
results_df = results_df.reindex(columns=models, level=0)
results_df
