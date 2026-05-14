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
# # Evaluating predicted tree cover against reference datasets
#
# Steps involved:
# 1. Collect Nick's tree-cover labels, restrict to recent tiles (year > 2017)
#    with 10–90 % tree cover, and re-attach the Köppen zone of each tile.
# 2. Crop each reference product to match these tiles
# 3. Sample the same pixels from each data source and compute precision / recall / accuracy per koppen zone.
#
# Need to have pre-downloaded all the worldcover tiles (I did this manually), and the meta canopy height tiles (I used ``demo_canopy_height_download.py``) on NCI before running this script.

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


# %%
# %%time

# v1:
# canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'
# canopy_height_geojson = 'tiles_global.geojson'
# canopy_height_folder = '/scratch/xe2/cb8590/Nick_GCH'

# v2:
canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height_v2'                                       # Initially empty directory
canopy_height_geojson = '/home/147/cb8590/Projects/shelterbelts/tiles_global.geojson'                  # Created this by running demo_canopy_height.py, and intersecting the resulting tiles_global.py with the aus boundary in QGIS. Note: need to make sure this matches the canopy_baseurl, since v1 used 60km tiles and v2 uses 30km tiles (will get a 404 error if it doesn't match).
canopy_height_folder = '/scratch/xe2/cb8590/Nick_GCH_v2'

# %%
gdf_nodesert = gpd.read_file(f'{nick_outlines}/footprints_large_recent_nodesert.gpkg')  # Created by barra_ag_crop.py
gdf_nodesert['stub'] = [f.split('.')[0] for f in gdf_nodesert['filename']]

gdf_outlines = gdf_nodesert.to_crs('EPSG:4326')
already_done = {f.replace('_canopy_height.tif', '') for f in os.listdir(canopy_height_folder) if f.endswith('_canopy_height.tif')}
gdf_outlines = gdf_outlines[~gdf_outlines['stub'].isin(already_done)]
print(f"{len(gdf_outlines)} tiles remaining")

# gdf_outlines = gdf_outlines[:2]


# %%
# # %%time
# # This took about 4 hours in a pbs job
# # save_crops(gdf_outlines, canopy_height_dir, canopy_height_geojson, canopy_height_folder, 'canopy_height', id_column='tile')

# %% [markdown]
# ## 3. Sample a jittered grid per tile
#
# For each label tile we load the label plus four predictions (WorldCover, GCH v1, GCH v2,
# my predictions), reproject-match them, and sample pixels on a jittered grid.
# Tiles missing any product are skipped silently.

# %%
canopy_height_folder_v1 = '/scratch/xe2/cb8590/Nick_GCH'
canopy_height_folder_v2 = '/scratch/xe2/cb8590/Nick_GCH_v2'

gdf_nodesert_joined = gdf_nodesert.to_crs('EPSG:3857').sjoin_nearest(
    gdf_koppen[['geometry', 'Name']], how='left', distance_col='distance_to_koppen')

# %%
# %%time
def create_df_evaluation(limit=None):
    rows = []
    gdf_eval = gdf_nodesert_joined if limit is None else gdf_nodesert_joined.iloc[:limit]
    for i, row in gdf_eval.iterrows():
        tile, koppen_class = row['stub'], row['Name']

        label_path = f'{nick_aus_treecover_10m}/{tile}.tiff'
        wc_path    = f'{worldcover_folder}/{tile}_worldcover.tif'
        gch1_path  = f'{canopy_height_folder_v1}/{tile}_canopy_height.tif'
        gch2_path  = f'{canopy_height_folder_v2}/{tile}_canopy_height.tif'
        pred_path  = f'{my_prediction_folder}/{tile}_my_prediction.tif'

        if not all(os.path.exists(p) for p in [label_path, wc_path, gch1_path, gch2_path, pred_path]):
            continue

        ds_label = rxr.open_rasterio(label_path).isel(band=0).drop_vars('band').astype(int)
        ds_wc = rxr.open_rasterio(wc_path).isel(band=0).drop_vars('band')
        ds_wc_trees = ((ds_wc == 10) | (ds_wc == 20)).astype(int)
        ds_gch1 = rxr.open_rasterio(gch1_path).isel(band=0).drop_vars('band')
        ds_gch1_trees = (ds_gch1 >= 1).astype(int)
        ds_gch2 = rxr.open_rasterio(gch2_path).isel(band=0).drop_vars('band')
        ds_gch2_trees = (ds_gch2 >= 1).astype(int)
        ds_pred = rxr.open_rasterio(pred_path).isel(band=0).drop_vars('band')
        ds_pred_trees = (ds_pred >= 50).astype(int)

        ds = xr.Dataset({
            'nick_trees':                    ds_label.rio.reproject_match(ds_pred_trees),
            'worldcover_trees':              ds_wc_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'global_canopy_height_v1_trees': ds_gch1_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'global_canopy_height_v2_trees': ds_gch2_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'my_predictions':                ds_pred_trees,
        })
        df = jittered_grid(ds, spacing=20)
        df['tile'] = tile
        df['koppen_class'] = koppen_class
        rows.append(df)

    df_all = pd.concat(rows)
    filename = f'{nick_outlines}/df_evaluation_nodesert.csv'
    df_all.to_csv(filename)
    print("Saved:",filename)
    return df_all

# Took 8 mins to do ~2k tiles and create ~100k rows
df_all = create_df_evaluation()
df_all

# %% [markdown]
# ## 4. Precision / recall / accuracy per Köppen zone

# %%
koppen_order = ['Aw', 'BSh', 'BSk', 'CFa', 'Cfb']
models = ['worldcover_trees', 'global_canopy_height_v1_trees', 'global_canopy_height_v2_trees', 'my_predictions']

rows = []
for koppen_class in koppen_order:
    df = df_all[df_all['koppen_class'] == koppen_class]
    if df.empty:
        continue
    row = {'koppen_class': koppen_class}
    for model in models:
        y_true, y_pred = df['nick_trees'], df[model]
        row[(model, 'precision')] = round(precision_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'recall')]    = round(recall_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'accuracy')]  = round(accuracy_score(y_true, y_pred), 2)
    rows.append(row)

results_df = pd.DataFrame(rows).set_index('koppen_class')
results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)
results_df = results_df.reindex(columns=models, level=0)
results_df

# %%
