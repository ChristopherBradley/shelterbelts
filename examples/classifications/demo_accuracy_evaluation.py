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
# gdf_outlines = gdf_outlines[:2]


# %%
# # %%time
# # This took about 4 hours in a pbs job
# # save_crops(gdf_outlines, canopy_height_dir, canopy_height_geojson, canopy_height_folder, 'canopy_height', id_column='tile')

# %%
# %%time

# Queensland Woody Extent 2023 (vector dataset rasterized to 10 m, EPSG:3577)
qld_woody_gdb    = '/scratch/xe2/cb8590/DP_QLD_WOODY_EXTENT_2023.gdb'
qld_woody_folder = '/scratch/xe2/cb8590/Nick_Queensland_Woody'

already_done_qld = {f.replace('_qld_woody.tif', '') for f in os.listdir(qld_woody_folder) if f.endswith('_qld_woody.tif')}
gdf_qld_remaining = gdf_nodesert.to_crs('EPSG:3577')
gdf_qld_remaining = gdf_qld_remaining[~gdf_qld_remaining['stub'].isin(already_done_qld)]
print(f"{len(gdf_qld_remaining)} QLD woody tiles remaining")


# %%
# gdf_qld_remaining = gdf_qld_remaining[:10]

# %%
# # %%time
def crop_qld_evaluation_data():
    # Rasterizes QLD woody extent polygons to 10 m GeoTIFFs (one per nodesert tile).
    # Tiles outside QLD produce no file and are skipped silently in the evaluation.
    # Ran as a PBS job — see pbs_scripts/accuracy_categories.pbs
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    os.makedirs(qld_woody_folder, exist_ok=True)
    for _, row in gdf_qld_remaining.iterrows():
        ref_path = os.path.join(nick_aus_treecover_10m, f"{row['stub']}.tiff")
        if not os.path.exists(ref_path):
            continue
        with rasterio.open(ref_path) as ref:
            ref_crs       = ref.crs
            ref_transform = ref.transform
            ref_shape     = (ref.height, ref.width)
            ref_bounds    = ref.bounds
        bbox_gdb = transform_bounds(ref_crs, 'EPSG:3577', *ref_bounds)
        gdf_tile = gpd.read_file(qld_woody_gdb, bbox=bbox_gdb)
        if gdf_tile.empty:
            continue
        gdf_woody = gdf_tile[gdf_tile['woody_extent_2023'] == 1].copy()
        gdf_woody = gdf_woody.to_crs(ref_crs)
        tile_geom = box(*ref_bounds)
        gdf_woody['geometry'] = gdf_woody['geometry'].intersection(tile_geom)
        gdf_woody = gdf_woody[~gdf_woody['geometry'].is_empty]
        shapes = [(geom, 1) for geom in gdf_woody.geometry if geom is not None and not geom.is_empty]
        burned = rasterize(shapes, out_shape=ref_shape, transform=ref_transform, fill=0, dtype='uint8')
        out_path = os.path.join(qld_woody_folder, f"{row['stub']}_qld_woody.tif")
        with rasterio.open(out_path, 'w', driver='GTiff', height=ref_shape[0], width=ref_shape[1],
                           count=1, dtype='uint8', crs=ref_crs, transform=ref_transform,
                           compress='lzw') as dst:
            dst.write(burned, 1)



# %% [markdown]
# ## 3. Sample a jittered grid per tile
#
# For each label tile we load the label plus four predictions (WorldCover, GCH v1, GCH v2,
# my predictions), reproject-match them, and sample pixels on a jittered grid.
# Tiles missing any product are skipped silently.

# %%
canopy_height_folder_v1 = '/scratch/xe2/cb8590/Nick_GCH'
canopy_height_folder_v2 = '/scratch/xe2/cb8590/Nick_GCH_v2'
# qld_woody_folder defined above

gdf_nodesert_joined = gdf_nodesert.to_crs('EPSG:3857').sjoin_nearest(
    gdf_koppen[['geometry', 'Name']], how='left', distance_col='distance_to_koppen')


# %%


# %%time
def create_df_evaluation(limit=None):
    rows = []
    gdf_eval = gdf_nodesert_joined if limit is None else gdf_nodesert_joined.iloc[:limit]
    for i, row in gdf_eval.iterrows():
        tile, koppen_class = row['stub'], row['Name']

        label_path    = f'{nick_aus_treecover_10m}/{tile}.tiff'
        wc_path       = f'{worldcover_folder}/{tile}_worldcover.tif'
        gch1_path     = f'{canopy_height_folder_v1}/{tile}_canopy_height.tif'
        gch2_path     = f'{canopy_height_folder_v2}/{tile}_canopy_height.tif'
        qld_path      = f'{qld_woody_folder}/{tile}_qld_woody.tif'
        pred_path     = f'{my_prediction_folder}/{tile}_my_prediction.tif'

        required = [label_path, wc_path, gch1_path, gch2_path, pred_path]
        if not all(os.path.exists(p) for p in required):
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

        ds_qld_trees = (
            (rxr.open_rasterio(qld_path).isel(band=0).drop_vars('band') >= 1)
            .astype(int)
            .rio.reproject_match(ds_pred_trees, resampling=Resampling.max)
            if os.path.exists(qld_path) else None
        )
        dataset = {
            'nick_trees':                    ds_label.rio.reproject_match(ds_pred_trees),
            'worldcover_trees':              ds_wc_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'global_canopy_height_v1_trees': ds_gch1_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'global_canopy_height_v2_trees': ds_gch2_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max),
            'my_predictions':                ds_pred_trees,
            **({'qld_woody_trees': ds_qld_trees} if ds_qld_trees is not None else {}),
        }

        ds = xr.Dataset(dataset)
        df = jittered_grid(ds, spacing=20)
        df['tile'] = tile
        df['koppen_class'] = koppen_class
        rows.append(df)

    df_all = pd.concat(rows)
    filename = f'{nick_outlines}/df_evaluation_nodesert.csv'
    df_all.to_csv(filename)
    print("Saved:", filename)
    return df_all

df_all = create_df_evaluation()
df_all


# %%
df_all = df_all.dropna()
df_all.to_csv(f'{nick_outlines}/df_evaluation_qld.csv')

# %% [markdown]
# ## 4. Precision / recall / accuracy per Köppen zone

# %%
koppen_order = ['Aw', 'BSh', 'BSk', 'CFa', 'Cfb']
# qld_woody_trees only present for tiles inside QLD — included when available
models = ['worldcover_trees', 'global_canopy_height_v1_trees', 'global_canopy_height_v2_trees', 'qld_woody_trees', 'my_predictions']

rows = []
for koppen_class in koppen_order:
    df = df_all[df_all['koppen_class'] == koppen_class]
    if df.empty:
        continue
    row = {'koppen_class': koppen_class}
    for model in models:
        if model not in df.columns:
            continue
        mask = df[model].isin([0, 1])
        y_true = df.loc[mask, 'nick_trees'].values
        y_pred = df.loc[mask, model].values
        row[(model, 'precision')] = round(precision_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'recall')]    = round(recall_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'accuracy')]  = round(accuracy_score(y_true, y_pred), 2)
    rows.append(row)

results_df = pd.DataFrame(rows).set_index('koppen_class')
results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)
results_df = results_df.reindex(columns=models, level=0)
results_df
