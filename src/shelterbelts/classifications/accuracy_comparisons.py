# +
# Compare my model in different koppen regions and years with worldcover and canopy height, using the tiles with 10%-90% tree cover
# -

import os
import glob
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import random
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage
from sklearn.metrics import accuracy_score, classification_report

from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds
from shelterbelts.apis.worldcover import tif_categorical, worldcover_cmap


# +
tmpdir = '/scratch/xe2/cb8590/tmp'

worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'  # Should move these to gdata so they don't disappear.
worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
worldcover_folder = '/scratch/xe2/cb8590/Nick_worldcover_reprojected'

my_prediction_dir = '/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/'
my_prediction_geojson = 'barra_trees_s4_aus_4326_weightings_median_2020_subfolders__footprints.gpkg'
my_prediction_folder = '/scratch/xe2/cb8590/Nick_2020_predicted'

# -



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

# !ls {my_prediction_dir}/{my_prediction_geojson}

gpd.GeoSeries(row['geometry'], crs="EPSG:3857").to_file('/scratch/xe2/cb8590/tmp/prediction_wont_merge.gpkg')

row = gdf_merged_3857.iloc[3]
bbox = row['geometry'].bounds
filename = row['filename']
stub = row['stub']
mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, filename, my_prediction_dir, my_prediction_geojson, 'filename', verbose=True) 


# +
# %%time
# Crop and save a tif for each of Nick's tiles
name = 'my_prediction'  # Also swapped my_prediction_dir, my_prediction_geojson, my_prediction_folder
for i, row in gdf_merged_3857[:10].iterrows():
    bbox_3857 = row['geometry'].bounds
    filename = row['filename']
    stub = row['stub']
    
    mosaic, out_meta = merge_tiles_bbox(bbox_3857, tmpdir, filename, my_prediction_dir, my_prediction_geojson, 'filename', verbose=False) 
    ds = merged_ds(mosaic, out_meta, name)
    da = ds[name].rename({'longitude':'x', 'latitude':'y'})
    out_filename = os.path.join(my_prediction_folder, f'{stub}_{name}.tif')
    tif_categorical(da, out_filename, worldcover_cmap)  # Should be using a different cmap

# Took 4 mins.
# -

np.random.seed(0)
random.seed(0)

def jittered_grid(ds):
    """Create a grid of coordinates spaced 100m apart, with a random 2 pixel jitter"""
    spacing = 10
    jitter_range = np.arange(-2, 3)  # [-2, -1, 0, 1, 2]

    # Create a coordinate grid
    edge_buffer = spacing//2
    y_inds = np.arange(edge_buffer, ds.sizes['y'] - edge_buffer, spacing)
    x_inds = np.arange(edge_buffer, ds.sizes['x'] - edge_buffer, spacing)

    # Apply jitter
    y_jittered_inds = y_inds + np.random.choice(jitter_range, size=len(y_inds))
    x_jittered_inds = x_inds + np.random.choice(jitter_range, size=len(x_inds))

    # Get actual coordinates
    y_jittered_coords = ds['y'].values[y_jittered_inds]
    x_jittered_coords = ds['x'].values[x_jittered_inds]

    # Get non-time-dependent variables
    aggregated_vars = [var for var in ds.data_vars if 'time' not in ds[var].dims]

    data_list = []
    for y_coord in y_jittered_coords:
        for x_coord in x_jittered_coords:
            values = {var: ds[var].sel(y=y_coord, x=x_coord, method='nearest').item() for var in aggregated_vars}
            values.update({'y': y_coord, 'x': x_coord})
            data_list.append(values)

        
    # Create DataFrame
    df = pd.DataFrame(data_list)
    return df


def tile_csv(tile):
    """Merge the trees from Nick's files and worldcover reprojected"""
    tile_id = "_".join(tile.split('/')[-1].split('_')[:2])

    tree_cover_filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{tile_id}_binary_tree_cover_10m.tiff'
    ds_nick_trees = rxr.open_rasterio(tree_cover_filename).isel(band=0).drop_vars('band').astype(int)

    worldcover_filename = f'/scratch/xe2/cb8590/Nick_worldcover_reprojected/{tile_id}_worldcover.tif'
    ds_worldcover = rxr.open_rasterio(worldcover_filename).isel(band=0).drop_vars('band')
    ds_worldcover_trees = ((ds_worldcover == 10) | (ds_worldcover == 20)).astype(int)  # 10 is trees, and 20 is shrubs
    
    gch_filename = f'/scratch/xe2/cb8590/Nick_GCH/{tile_id}_10m.tif'
    ds_gch = rxr.open_rasterio(gch_filename).isel(band=0).drop_vars('band')
    ds_gch_trees = (ds_gch >= 1).astype(int)

    ds = xr.Dataset({
        "nick_trees": ds_nick_trees,
        "worldcover_trees": ds_worldcover_trees,
        "global_canopy_height_trees": ds_gch_trees
    })

    df = jittered_grid(ds)
    df["tile_id"] = tile_id
    
    return df

# +
# # Create a dataframe of imagery and woody veg or canopy cover classifications for each tile
# tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
# world_cover_dir = "/scratch/xe2/cb8590/Nick_worldcover_reprojected"
# global_canopy_height_dir = "/scratch/xe2/cb8590/Nick_GCH"

# # outdir = "/scratch/xe2/cb8590/Nick_csv"

# # Using the sentinel tiles since these were used to train my model, so I can compare worldcover on the same tiles
# sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
# sentinel_tiles = glob.glob(f'{sentinel_dir}/*')
# print("Number of sentinel tiles:", len(sentinel_tiles))

# # Using the GCH tiles because I haven't finished downloading them, so this just uses the ones already downloaded
# gch_tiles = glob.glob(f'{global_canopy_height_dir}/*10m.tif')
# print("Number of global canopy height tiles:", len(gch_tiles))

# # Using the GCH tiles because I haven't finished downloading them, so this just uses the ones already downloaded
# worldcover_tiles = glob.glob(f'{world_cover_dir}/*')
# print("Number of worldcover tiles:", len(worldcover_tiles))

# # Random sample of 1000 tiles so it doesn't take so long to read them all in
# sampled_tiles = random.sample(gch_tiles, 1000)
# -



# +
# # %%time
# # Find the tree status from worldcover, global canopy height and Nick's tiff files for lots of jittered gridded points across lots of files
# def nick_vs_gch():
#     """Run the tile_csv function on all the tiles"""
#     dfs = []
#     for tile in sampled_tiles:
#         df = tile_csv(tile)
#         dfs.append(df)

#     df_all = pd.concat(dfs)
    
#     # Save these tree comparisons
#     filename = "/g/data/xe2/cb8590/Nick_outlines/nick_vs_gch.csv"
#     df_all.to_csv(filename, index="False")

# # Took 30 secs for 100, 3 mins for 700, 6 mins for 1000

# # Load these comparisons we've just saved
# filename = "/g/data/xe2/cb8590/Nick_outlines/nick_vs_gch.csv"
# df_all = pd.read_csv(filename)

# df_all['global_canopy_height_trees'].value_counts()

# print(classification_report(df_all['nick_trees'], df_all['worldcover_trees']))

# # Worldcover has an accuracy of 82% but recall for 1's of only 55% without shrubs or 62% with shrubs. So it misses a lot of trees, as I've noticed qualitatively before.

# print(classification_report(df_all['nick_trees'], df_all['global_canopy_height_trees']))

# # Global canopy height has an accuracy of 83% with a recall of 65% for 1's. So a little better than worldcover

# # Load the bioregions per tile generated in megaregions.py
# filename_bioregions = "/g/data/xe2/cb8590/Nick_outlines/centroids_named.gpkg"
# gdf_bioregions = gpd.read_file(filename_bioregions)
# gdf_bioregions["tile_id"] = ["_".join(tile.split('/')[-1].split('_')[:2]) for tile in gdf_bioregions['filename']]

# # Add the bioregions to the classifications
# df = df_all.merge(gdf_bioregions)[['nick_trees', 'worldcover_trees', 'global_canopy_height_trees', 'Name', 'Full Name']]
# df = df.rename(columns={'Full Name':'koppen_class'})

# # # Print a classification report for each bioregion
# # for bioregion in df['koppen_class'].unique():
# #     df_bioregion = df[df['koppen_class'] == bioregion]
# #     print(bioregion)
# #     print("Worldcover")
# #     print(classification_report(df_bioregion['nick_trees'], df_bioregion['worldcover_trees']))
# #     print("Global Canopy Height")
# #     print(classification_report(df_bioregion['nick_trees'], df_bioregion['global_canopy_height_trees']))

# # Create a classification report for each koppen category and tree class
# worldcover_rows = []
# canopy_rows = []

# # Loop through each bioregion
# for bioregion in df['koppen_class'].unique():
#     df_bioregion = df[df['koppen_class'] == bioregion]
    
#     for tree_class in [0.0, 1.0]:
#         # Worldcover metrics
#         report_wc = classification_report(
#             df_bioregion['nick_trees'], 
#             df_bioregion['worldcover_trees'], 
#             output_dict=True, 
#             zero_division=0
#         )
#         accuracy_wc = accuracy_score(df_bioregion['nick_trees'], df_bioregion['worldcover_trees'])

#         if str(tree_class) in report_wc:
#             worldcover_rows.append({
#                 'koppen_class': bioregion,
#                 'tree_class': tree_class,
#                 'precision': report_wc[str(tree_class)]['precision'],
#                 'recall': report_wc[str(tree_class)]['recall'],
#                 'f1-score': report_wc[str(tree_class)]['f1-score'],
#                 'accuracy': accuracy_wc,
#                 'support': report_wc[str(tree_class)]['support'],
#             })
#         worldcover_table = pd.DataFrame(worldcover_rows)
            
#         # Overall worldcover metrics 
#         overall_report = classification_report(df['nick_trees'], df['worldcover_trees'], output_dict=True, zero_division=0)
#         overall_accuracy = accuracy_score(df['nick_trees'], df['worldcover_trees'])

#         for tree_class in [0.0, 1.0]:
#             if str(tree_class) in overall_report:
#                 worldcover_table.loc[len(worldcover_table)] = {
#                     'koppen_class': 'overall',
#                     'tree_class': tree_class,
#                     'precision': overall_report[str(tree_class)]['precision'],
#                     'recall': overall_report[str(tree_class)]['recall'],
#                     'f1-score': overall_report[str(tree_class)]['f1-score'],
#                     'accuracy': overall_accuracy,
#                     'support': overall_report[str(tree_class)]['support'],
#                 }

#         # Global Canopy Height metrics
#         report_gch = classification_report(
#             df_bioregion['nick_trees'], 
#             df_bioregion['global_canopy_height_trees'], 
#             output_dict=True, 
#             zero_division=0
#         )
#         accuracy_gch = accuracy_score(df_bioregion['nick_trees'], df_bioregion['global_canopy_height_trees'])

#         if str(tree_class) in report_gch:
#             canopy_rows.append({
#                 'koppen_class': bioregion,
#                 'tree_class': tree_class,
#                 'precision': report_gch[str(tree_class)]['precision'],
#                 'recall': report_gch[str(tree_class)]['recall'],
#                 'f1-score': report_gch[str(tree_class)]['f1-score'],
#                 'accuracy': accuracy_gch,
#                 'support': report_gch[str(tree_class)]['support'],
#             })
#         canopy_height_table = pd.DataFrame(canopy_rows)
        
#         # Overall global canopy height metrics
#         overall_report = classification_report(df['nick_trees'], df['global_canopy_height_trees'], output_dict=True, zero_division=0)
#         overall_accuracy = accuracy_score(df['nick_trees'], df['global_canopy_height_trees'])

#         for tree_class in [0.0, 1.0]:
#             if str(tree_class) in overall_report:
#                 canopy_height_table.loc[len(canopy_height_table)] = {
#                     'koppen_class': 'overall',
#                     'tree_class': tree_class,
#                     'precision': overall_report[str(tree_class)]['precision'],
#                     'recall': overall_report[str(tree_class)]['recall'],
#                     'f1-score': overall_report[str(tree_class)]['f1-score'],
#                     'accuracy': overall_accuracy,
#                     'support': overall_report[str(tree_class)]['support'],
#                 }

# worldcover_table

# canopy_height_table
# -


