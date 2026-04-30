# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluating predicted tree cover against reference datasets
#
# This notebook compares our trained neural-network predictions against SA WorldCover and Meta Global
# Canopy Height by Köppen climate zone.
#
# 1. Collect Nick's tree-cover labels, restrict to recent tiles (year > 2017)
#    with 10–90 % tree cover, and re-attach the Köppen zone of each tile.
# 2. Crop each reference product to match these tiles
# 3. Sample the same pixels from each data source and compute precision / recall / accuracy per koppen zone.
#
# All inputs live under ``/g/data/xe2/cb8590/...`` on NCI.
# Need to have downloaded all the worldcover tiles (I did this manually), and the meta (I used ``canopy_height_download.py``)

# %%
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from sklearn.metrics import accuracy_score, precision_score, recall_score

from shelterbelts.apis.worldcover import worldcover_cmap
from shelterbelts.classifications.merge_inputs_outputs import jittered_grid
from shelterbelts.utils.filepaths import (
    canopy_height_dir,
    canopy_height_folder,
    canopy_height_geojson,
    koppen_australia,
    my_prediction_dir,
    my_prediction_folder,
    my_prediction_geojson,
    nick_aus_treecover_10m,
    nick_outlines,
    tmpdir,
    worldcover_dir,
    worldcover_folder,
    worldcover_geojson,
)
from shelterbelts.utils.tiles import merge_tiles_bbox, merged_ds
from shelterbelts.utils.visualisation import tif_categorical

np.random.seed(0)

# %% [markdown]
# ## 1. Build the tile catalogue with Köppen zones attached

# %%
gdf_years = gpd.read_file(f'{nick_outlines}/tiff_footprints_years.gpkg')
gdf_percent = gpd.read_file(f'{nick_aus_treecover_10m}/cb8590_Nick_Aus_treecover_10m_footprints.gpkg')

gdf_good = gdf_percent[~gdf_percent['bad_tif']].drop(columns='geometry')
gdf_recent = gdf_years[gdf_years['year'] > 2017]

# Two reprojected copies — 3857 for merging Sentinel / predictions, 4326 for WorldCover.
gdf_merged_4326 = gdf_good.merge(gdf_recent.to_crs('EPSG:4326'), how='inner', on='filename')
gdf_merged_3857 = gdf_good.merge(gdf_recent.to_crs('EPSG:3857'), how='inner', on='filename')
for gdf in (gdf_merged_4326, gdf_merged_3857):
    gdf['stub'] = [f.split('.')[0] for f in gdf['filename']]

gdf_merged_3857 = gpd.GeoDataFrame(gdf_merged_3857, geometry=gdf_merged_3857['geometry'],
                                   crs='EPSG:3857')

gdf_koppen = gpd.read_file(koppen_australia).to_crs('EPSG:3857')
gdf_joined = gdf_merged_3857.sjoin_nearest(
    gdf_koppen[['geometry', 'Name']], how='left', distance_col='distance_to_koppen')

# %% [markdown]
# ## 2. Crop the reference products to each tile

# %%
def save_crops(gdf_tiles, source_dir, source_geojson, out_folder, name, id_column='filename'):
    """Crop ``source_dir`` tiles to each tile footprint in ``gdf_tiles`` and write a tif."""
    os.makedirs(out_folder, exist_ok=True)
    for _, row in gdf_tiles.iterrows():
        bbox = row['geometry'].bounds
        mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, row['filename'], source_dir,
                                            source_geojson, id_column, verbose=False)
        da = merged_ds(mosaic, out_meta, name)[name].rename({'longitude': 'x', 'latitude': 'y'})
        tif_categorical(da, os.path.join(out_folder, f"{row['stub']}_{name}.tif"), worldcover_cmap)


# save_crops(gdf_merged_4326, worldcover_dir, worldcover_geojson, worldcover_folder, 'worldcover')
# save_crops(gdf_merged_3857, my_prediction_dir, my_prediction_geojson, my_prediction_folder, 'my_prediction')
# save_crops(gdf_merged_4326, canopy_height_dir, canopy_height_geojson, canopy_height_folder, 'canopy_height', id_column='tile')

# %% [markdown]
# ## 3. Sample a jittered grid per tile
#
# For each label tile we load the four rasters (label + three predictions),
# reproject-match them, and sample pixels.

# %%
def create_df_evaluation(limit=5000):
    rows = []
    for i, row in gdf_joined.iloc[:limit].iterrows():
        tile, koppen_class = row['stub'], row['Name']

        ds_label = rxr.open_rasterio(f'{nick_aus_treecover_10m}/{tile}.tiff').isel(band=0).drop_vars('band').astype(int)
        ds_wc = rxr.open_rasterio(f'{worldcover_folder}/{tile}_worldcover.tif').isel(band=0).drop_vars('band')
        ds_wc_trees = ((ds_wc == 10) | (ds_wc == 20)).astype(int)
        ds_gch = rxr.open_rasterio(f'{canopy_height_folder}/{tile}_canopy_height.tif').isel(band=0).drop_vars('band')
        ds_gch_trees = (ds_gch >= 1).astype(int)
        ds_pred = rxr.open_rasterio(f'{my_prediction_folder}/{tile}_my_prediction.tif').isel(band=0).drop_vars('band')
        ds_pred_trees = (ds_pred >= 50).astype(int)

        ds = xr.Dataset({
            'nick_trees': ds_label.rio.reproject_match(ds_pred_trees),
            'worldcover_trees': ds_wc_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'global_canopy_height_trees': ds_gch_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'my_predictions': ds_pred_trees,
        })
        df = jittered_grid(ds, spacing=20)
        df['tile'] = tile
        df['koppen_class'] = koppen_class
        rows.append(df)

    df_all = pd.concat(rows)
    df_all.to_csv(f'{nick_outlines}/df_evaluation_10pct-90pct.csv')
    return df_all


# df_all = create_df_evaluation()
df_all = pd.read_csv(f'{nick_outlines}/df_evaluation_10pct-90pct.csv')

# %% [markdown]
# ## 4. Precision / recall / accuracy per Köppen zone

# %%
koppen_order = ['Aw', 'BSh', 'BWh', 'BSk', 'CFa', 'Cfb']
models = ['worldcover_trees', 'global_canopy_height_trees', 'my_predictions']

rows = []
for koppen_class in koppen_order:
    df = df_all[df_all['koppen_class'] == koppen_class]
    if df.empty:
        continue
    row = {'koppen_class': koppen_class}
    for model in models:
        y_true, y_pred = df['nick_trees'], df[model]
        row[(model, 'precision')] = round(precision_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'recall')] = round(recall_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'accuracy')] = round(accuracy_score(y_true, y_pred), 2)
    rows.append(row)

results_df = pd.DataFrame(rows).set_index('koppen_class')
results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)
results_df = results_df.reindex(columns=models, level=0)
results_df
