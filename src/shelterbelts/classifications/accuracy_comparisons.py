# +
# Compare my model in different koppen regions and years with worldcover and canopy height, using the tiles with 10%-90% tree cover
# -

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import random
import rioxarray as rxr
import xarray as xr
from sklearn.metrics import accuracy_score, precision_score, recall_score


from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds
from shelterbelts.apis.worldcover import tif_categorical, worldcover_cmap
from shelterbelts.classifications.merge_inputs_outputs import jittered_grid


np.random.seed(0)
random.seed(0)

# +
tmpdir = '/scratch/xe2/cb8590/tmp'

worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'  # Should move these to gdata so they don't disappear.
worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
worldcover_folder = '/scratch/xe2/cb8590/Nick_worldcover_reprojected'

my_prediction_dir = '/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/'
my_prediction_geojson = 'barra_trees_s4_aus_4326_weightings_median_2020_subfolders__footprints.gpkg'
my_prediction_folder = '/scratch/xe2/cb8590/Nick_2020_predicted'

canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'
canopy_height_geojson = 'tiles_global.geojson'
canopy_height_folder = '/scratch/xe2/cb8590/Nick_GCH'

# +
# Load Nick's raster footprints
filename = '/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg'
gdf_years = gpd.read_file(filename)

# I don't trust the geometry of the current gdf_percent because it was using a utm crs, when the tiles span many utms. 
gdf_percent = gpd.read_file('/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg')

# Taking the useful features from both gpkgs
gdf_good = gdf_percent[~gdf_percent['bad_tif']].drop(columns='geometry')
gdf_recent = gdf_years[gdf_years['year'] > 2017]
gdf_recent_4326 = gdf_recent.to_crs("EPSG:4326")
gdf_merged_4326 = gdf_good.merge(gdf_recent_4326, how='inner', on='filename')
gdf_merged_4326['stub'] = [f.split('.')[0] for f in gdf_merged_4326['filename']]

gdf_recent_3857 = gdf_recent.to_crs("EPSG:3857")
gdf_merged_3857 = gdf_good.merge(gdf_recent_3857, how='inner', on='filename')
gdf_merged_3857['stub'] = [f.split('.')[0] for f in gdf_merged_3857['filename']]

# Convert back to geodataframe
gdf_merged_3857 = gpd.GeoDataFrame(
    gdf_merged_3857,
    geometry=gdf_recent_3857.geometry.name,
    crs=gdf_recent_3857.crs
)

# Visualising the centroids
gdf_centroids = gdf_merged_3857.copy()
gdf_centroids['geometry'] = gdf_merged_3857.geometry.centroid
# gdf_centroids.to_file('/scratch/xe2/cb8590/tmp/Nick_training_centroids.gpkg')
# -
# Attach the koppen classes
gdf_koppen = gpd.read_file('/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned2.gpkg')
gdf_koppen_3857 = gdf_koppen.to_crs("EPSG:3857")
gdf_joined = gdf_merged_3857.sjoin_nearest(
    gdf_koppen_3857[['geometry', 'Name']],
    how="left",
    distance_col="distance_to_koppen"
)


gdf_joined


# +

# %%time
def save_nick_worldcover():
    # Crop and save a tif for each of Nick's tiles
    for i, row in gdf_merged_4326.iterrows():
        bbox_4326 = row['geometry'].bounds
        filename = row['filename']
        stub = row['stub']
        
        mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, filename, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
        ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
        da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
        worldcover_filename = os.path.join(worldcover_folder, f'{stub}_worldcover.tif')
        tif_categorical(da_worldcover, worldcover_filename, worldcover_cmap)
    
    # Took 4 mins.


# -

# %%time
def save_nick_predicted():
    # Crop and save a tif for each of Nick's tiles
    name = 'my_prediction'  # Also swapped my_prediction_dir, my_prediction_geojson, my_prediction_folder
    for i, row in gdf_merged_3857.iterrows():
        bbox_3857 = row['geometry'].bounds
        filename = row['filename']
        stub = row['stub']
        
        mosaic, out_meta = merge_tiles_bbox(bbox_3857, tmpdir, filename, my_prediction_dir, my_prediction_geojson, 'filename', verbose=False) 
        ds = merged_ds(mosaic, out_meta, name)
        da = ds[name].rename({'longitude':'x', 'latitude':'y'})
        out_filename = os.path.join(my_prediction_folder, f'{stub}_{name}.tif')
        tif_categorical(da, out_filename, worldcover_cmap)  # Should be using a different cmap
    
    # Took 30 mins


# %%time
def save_nick_canopyheight():
    name = 'canopy_height'  
    my_prediction_dir = canopy_height_dir
    my_prediction_geojson = canopy_height_geojson
    my_prediction_folder = canopy_height_folder
    for i, row in gdf_merged_4326.iterrows():
        bbox = row['geometry'].bounds
        filename = row['filename']
        stub = row['stub']
        
        mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, filename, my_prediction_dir, my_prediction_geojson, 'tile', verbose=False) 
        ds = merged_ds(mosaic, out_meta, name)
        da = ds[name].rename({'longitude':'x', 'latitude':'y'})
        out_filename = os.path.join(my_prediction_folder, f'{stub}_{name}.tif')
        tif_categorical(da, out_filename, worldcover_cmap)  # Should be using a different cmap


# +
# %%time
from rasterio.enums import Resampling
def create_df_evaluation():
    limit = 5000
    gdf_subset = gdf_joined[:limit]
    dfs = []
    for i, row in gdf_subset.iterrows():
        if i % 100 == 0:
            print(f"Working on {i}/{len(gdf_subset)}")
        tile = row['stub']
        koppen_class = row['Name']
        tree_cover_filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{tile}.tiff'
        ds_nick_trees = rxr.open_rasterio(tree_cover_filename).isel(band=0).drop_vars('band').astype(int)
        
        worldcover_filename = f'/scratch/xe2/cb8590/Nick_worldcover_reprojected/{tile}_worldcover.tif'
        ds_worldcover = rxr.open_rasterio(worldcover_filename).isel(band=0).drop_vars('band')
        ds_worldcover_trees = ((ds_worldcover == 10) | (ds_worldcover == 20)).astype(int)  # 10 is trees, and 20 is shrubs
        
        gch_filename = f'/scratch/xe2/cb8590/Nick_GCH/{tile}_canopy_height.tif'
        ds_gch = rxr.open_rasterio(gch_filename).isel(band=0).drop_vars('band')
        ds_gch_trees = (ds_gch >= 1).astype(int)
        
        prediction_filename = f'/scratch/xe2/cb8590/Nick_2020_predicted/{tile}_my_prediction.tif'
        ds_pred = rxr.open_rasterio(prediction_filename).isel(band=0).drop_vars('band')
        ds_pred_trees = (ds_pred >= 50).astype(int)
        
        ds_nick_reprojected = ds_nick_trees.rio.reproject_match(ds_pred_trees)
        ds_worldcover_reprojected = ds_worldcover_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max)
        ds_gch_reprojected = ds_gch_trees.rio.reproject_match(ds_pred_trees, resampling=Resampling.max)
        
        ds = xr.Dataset({
            "nick_trees": ds_nick_reprojected,
            "worldcover_trees": ds_worldcover_reprojected,
            "global_canopy_height_trees": ds_gch_reprojected,
            "my_predictions": ds_pred_trees
        })
        
        df = jittered_grid(ds, spacing=20)
        df["tile"] = tile
        df["koppen_class"] = koppen_class
        dfs.append(df)
    
    df_all = pd.concat(dfs)
    filename = f'/g/data/xe2/cb8590/Nick_outlines/df_evaluation_10%-90%_2017-2024_resamplingmax.csv'
    df_all.to_csv(filename)
    # Took 15 mins
    
create_df_evaluation()
# -

# df_all = pd.read_csv('/g/data/xe2/cb8590/Nick_outlines/df_evaluation_10%-90%_2017-2024_limit5000.csv')
df_all = pd.read_csv('/g/data/xe2/cb8590/Nick_outlines/df_evaluation_10%-90%_2017-2024_resamplingmax.csv')

len(df_all)

df_all

# +

# Desired custom order
koppen_order = ['Aw', 'BSh', 'BWh', 'BSk', 'CFa', 'Cfb']
models = ['worldcover_trees', 'global_canopy_height_trees', 'my_predictions']
metrics = ['precision', 'recall', 'accuracy']

rows = []

for koppen_class in koppen_order:
    df = df_all[df_all['koppen_class'] == koppen_class]

    if len(df) == 0:
        continue  # skip missing classes

    row = {'koppen_class': koppen_class}

    for model in models:
        y_true = df['nick_trees']
        y_pred = df[model]

        row[(model, 'precision')] = round(precision_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'recall')]    = round(recall_score(y_true, y_pred, zero_division=0), 2)
        row[(model, 'accuracy')]  = round(accuracy_score(y_true, y_pred), 2)

    rows.append(row)

# Create MultiIndex DataFrame
results_df = pd.DataFrame(rows)
results_df = results_df.set_index('koppen_class')
results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)

# Ensure model order
results_df = results_df.reindex(columns=models, level=0)

results_df

