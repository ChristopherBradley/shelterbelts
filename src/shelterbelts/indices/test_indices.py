# +
import os
import glob
import argparse
import math
import pathlib

import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box

# Trying to avoid memory issues
import gc
import psutil
import resource
import subprocess, sys

# + endofcell="--"
from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.worldcover import tif_categorical
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds
from shelterbelts.apis.barra_daily import barra_daily

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import patch_metrics, linear_categories_cmap

# 11 secs for all these imports
# -
from shelterbelts.indices.opportunities import worldcover_dir, worldcover_geojson, hydrolines_gdb, roads_gdb  # Should make sure no other files import from this one, to avoid circular imports
# --

# # !pip install 'jupyterlab-lsp' 'python-lsp-server[all]'
# # !jupyter labextension install @krassowski/jupyterlab-lsp


from shelterbelts.indices.full_pipelines import run_pipeline_tifs, run_pipeline_tif, run_pipeline_csv
cover_threshold=50
min_patch_size=20
min_core_size=1000
edge_size=10
max_gap_size=1
distance_threshold=10
density_threshold=5 
buffer_width=3
strict_core_area=True
param_stub = ""
wind_method=None
wind_threshold=15
crop_pixels = 20
# crop_pixels = 0
limit = None
min_shelterbelt_length=20
max_shelterbelt_width=4
# folder = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717840/uint8_percentcover_res10_height2m/'
# outdir = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717840/linear_tifs'
# 
# folder='/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_28_lon_142'
folder = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/subfolders/lat_34_lon_148'
tmpdir = '/scratch/xe2/cb8590/tmp'
outdir=tmpdir
stub="TEST2"

# # %%time
# # percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/34_09-149_14_y2018_predicted_expanded20.tif'  # Exceeding memory
# percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/34_89-148_94_y2018_predicted_expanded20.tif' # Failing in pbs
# run_pipeline_tif(percent_tif, outdir=tmpdir, tmpdir=tmpdir, cover_threshold=50, crop_pixels=20, distance_threshold=40, buffer_width=10)



# +
# run_pipeline_tif(percent_tif, outdir=tmpdir, tmpdir=tmpdir, stub=None, 
#                      wind_method=None, wind_threshold=15,
#                      cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
#                      distance_threshold=10, density_threshold=5, buffer_width=3, strict_core_area=True,
#                      crop_pixels=0)

# +
# folder = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148'
# run_pipeline_tifs(folder, outdir=tmpdir, tmpdir=tmpdir, cover_threshold=50, crop_pixels=20, limit=10)

# +
# # %%time
# tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_26_y2018_predicted_expanded20.tif'
# run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20)
# -

# percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_22_y2018_predicted_expanded20.tif'
percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_26_y2018_predicted_expanded20.tif'


# +
# %%time
if stub is None:
    # stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'
    stub = percent_tif.split('/')[-1].split('.')[0][:50] # Hopefully there's something unique in the first 50 characters
data_folder = percent_tif[percent_tif.find('DATA'):percent_tif.find('DATA') + 11]

da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
da_trees = da_percent > cover_threshold

gs_bounds = gpd.GeoSeries([box(*da_percent.rio.bounds())], crs=da_percent.rio.crs)
bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])

# import pdb; pdb.set_trace()

worldcover_stub = f'{data_folder}_{stub}_{wind_method}_w{wind_threshold}_c{cover_threshold}_m{min_patch_size}_e{edge_size}_g{max_gap_size}_di{distance_threshold}_de{density_threshold}_b{buffer_width}' # Anything that might be run in parallel needs a unique filename, so we don't get rasterio merge conflicts
mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, worldcover_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=True) 
ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
gdf_hydrolines, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent)
gdf_roads, ds_roads = hydrolines(None, roads_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent, layer='NationalRoads_2025_09')

if wind_method and wind_method != "None":  # Handling conversion of None to "None" when using subprocess
    lat = (bbox_4326[1] + bbox_4326[3])/2
    lon = (bbox_4326[0] + bbox_4326[2])/2
    ds_wind = barra_daily(lat=lat, lon=lon, start_year=2020, end_year=2020, gdata=True, plot=False, save_netcdf=False) # This line is currently the limiting factor since it takes 4 secs
else:
    # if no wind_method provided than the percent_cover method without wind gets used
    ds_wind = None

ds_woody_veg = da_trees.to_dataset(name='woody_veg')
ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=min_patch_size, min_core_size=min_core_size, edge_size=edge_size, max_gap_size=max_gap_size, strict_core_area=strict_core_area, save_tif=False, plot=False, ds=ds_woody_veg)
ds_shelter = shelter_categories(None, wind_method=wind_method, wind_threshold=wind_threshold, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories, ds_wind=ds_wind, crop_pixels=crop_pixels)
ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=True, plot=False, da_worldcover=da_worldcover)

ds_buffer = buffer_categories(None, None, buffer_width=buffer_width, outdir=outdir, stub=stub, savetif=True, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines, ds_roads=ds_roads)
ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=True, crop_pixels=crop_pixels, min_shelterbelt_length=min_shelterbelt_length, max_shelterbelt_width=max_shelterbelt_width) 
# -


