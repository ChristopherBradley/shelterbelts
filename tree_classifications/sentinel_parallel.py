# Parallelise the sentinel downloads using a single job instead of hammering the scheduler with lots of jobs from a bash script

# +
import argparse
import os
import sys
import psutil
import ast
import glob

import math
import csv
import pickle
import numpy as np
import xarray as xr
import rioxarray
import datacube
import hdstats
import pandas as pd
import geopandas as gpd
from shapely.ops import transform
from pyproj import Transformer
import rasterio

from dea_tools.temporal import xr_phenology, temporal_statistics
from dea_tools.datahandling import load_ard
from dea_tools.bandindices import calculate_indices
from dea_tools.plotting import display_map, rgb
from dea_tools.dask import create_local_dask_cluster
from shapely.geometry import box
from pyproj import CRS, Transformer
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

import networkx as nx
import logging
import warnings
import contextlib
import io
import traceback, sys
warnings.filterwarnings('ignore')



# -

# Specific range so I can match Nick's tiff files
def define_query_range(lat_range, lon_range, time_range, input_crs='epsg:4326', output_crs='epsg:6933'):

    query = {
        'y': lat_range,
        'x': lon_range,
        'time': time_range,
        'resolution': (-10, 10),
        'crs':input_crs,  # Input CRS
        'output_crs': output_crs, # Output CRS
        'group_by': 'solar_day'
    }
    return query

def load_and_process_data(dc, query):
    # Silence the print statements
    with contextlib.redirect_stdout(io.StringIO()):
        ds = load_ard(
            dc=dc,
            products=['ga_s2am_ard_3', 'ga_s2bm_ard_3'],
            cloud_mask='s2cloudless',
            min_gooddata=0.9,
            measurements=['nbart_blue', 'nbart_green', 'nbart_red', 
                          'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
                          'nbart_nir_1', 'nbart_nir_2',
                          'nbart_swir_2', 'nbart_swir_3'],
            **query
        )
    return ds

# +
def sentinel_download(stub, year, outdir, bounds, src_crs):
    
    # print(f"Starting sentinel_download: {tif_id}, {year}\n")

    # Prep the DEA query
    lat_range = (bounds[1], bounds[3])
    lon_range = (bounds[0], bounds[2])
    time_range = (f"{year}-01-01", f"{year}-12-31")
    input_crs=src_crs 
    output_crs=src_crs
    query = define_query_range(lat_range, lon_range, time_range, input_crs, output_crs)

    # Load the data
    dc = datacube.Datacube(app=stub)
    ds = load_and_process_data(dc, query)

#     # Save the data
#     # If I'm running across all of Australia then I probably don't have space to save every tile as a pickle fie
#     filename = os.path.join(outdir, f'{tif_id}_ds2_{year}.pkl')
#     with open(filename, 'wb') as handle:
#         pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"Saved {filename}", flush=True)
    
    return ds


# -

def run_download(batch):
    # Get each worker to download a bunch of sentinel tiles based on Nick's tiff outlines
    for row in batch:
        tif, year, bounds, crs = row
        try:
            bbox = ast.literal_eval(bounds) # The bbox is saved as a string initially because I stored it in a csv file with pandas
            print(f"Downloading: {tif}_{year}", flush=True)
            stub = '_'.join(tif.split('_')[:2])
            sentinel_download(stub, year, outdir, bbox, crs)
        except Exception as e:
            print(f"Error in downloading: {tif}_{year}:", flush=True)
            traceback.print_exc(file=sys.stdout)

# %%time
def extract_bbox_year():
    """I used this code to extract the bbox and year for each of Nick's tiles"""
    # Read in the bbox and year for each of the tif files
    df = pd.read_csv(filename_gdf_maxyear, index_col='filename')

    # Load the crs and bounds for each tif file
    rows = list(df[['year']].itertuples(name=None))
    rows2 = []
    for row in rows:
        tif, year = row
        filename = os.path.join(indir, tif)
        with rasterio.open(filename) as src:
            bounds = src.bounds
            crs = src.crs.to_string()
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        rows2.append((tif, year, bbox, crs))

    df = pd.DataFrame(rows2, columns=["tif", "year", "bbox", "crs"])
    df.to_csv(filename_bbox_year, index=False)
    print("Saved", filename_bbox_year)

    # Took 3 mins


# Create rows for each of the 7k bbox's for tiffs Nick provided after 2017
def prep_rows_Nick():   
    indir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
    outdir = '/scratch/xe2/cb8590/Nick_sentinel'
    filename_bbox_year = "/g/data/xe2/cb8590/Nick_outlines/nick_bbox_year_crs.csv"
    filename_gdf_maxyear = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
    outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"

    # Load the tiff bboxs
    df = pd.read_csv(filename_bbox_year)
    df_2017_2022 = df[df['year'] >= 2017]
    print("Number of tiles between 2017-2022", len(df_2017_2022))

    # Find the tiles we have already downloaded
    sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
    sentinel_tiles = glob.glob(f'{sentinel_dir}/*')
    print("Number of sentinel tiles downloaded:", len(sentinel_tiles))

    # Find the tiles we haven't downloaded yet
    sentinel_tile_ids = ["_".join(sentinel_tile.split('/')[-1].split('_')[:2]) for sentinel_tile in sentinel_tiles]
    downloaded = [f"{tile_id}_binary_tree_cover_10m.tiff" for tile_id in sentinel_tile_ids]
    df_new = df_2017_2022[~df_2017_2022['tif'].isin(downloaded)]
    print("Number of new tiles to download:", len(df_new))

    rows = df_new[['tif', 'year', 'bbox', 'crs']].values.tolist()
    return rows


outdir = '/scratch/xe2/cb8590/ka08_trees'


# Load the sentinel imagery tiles
filename_sentinel_bboxs = "/g/data/xe2/cb8590/Nick_outlines/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
gdf = gpd.read_file(filename_sentinel_bboxs)

test_tile_id = "55HFC"
test_tile = gdf.loc[gdf['Name'] == test_tile_id, 'geometry'].values[0]

polygon = test_tile

test_tile_ids = "55HFC", "55HFD", "55HFE", "55HFB", "55HED", "55HDE", "55HCB"


# Having a look at the size of some tiles
for test_tile_id in test_tile_ids:
    polygon = gdf.loc[gdf['Name'] == test_tile_id, 'geometry'].values[0]

    # Get bounds of Sentinel polygon in degrees
    minx, miny, maxx, maxy = polygon.bounds
    width_deg = maxx - minx
    height_deg = maxy - miny
    print(f"Width in degrees: {width_deg}")
    print(f"Height in degrees: {height_deg}")

    # Determine UTM zone
    centroid = polygon.centroid
    lon, lat = centroid.x, centroid.y
    zone_number = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    utm_crs = f"+proj=utm +zone={zone_number} +datum=WGS84 +units=m +no_defs"
    if hemisphere == 'south':
        utm_crs += " +south"

    # Get bounds in metres
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    projected_polygon = transform(transformer.transform, polygon)
    minx_m, miny_m, maxx_m, maxy_m = projected_polygon.bounds
    width_m = maxx_m - minx_m
    height_m = maxy_m - miny_m
    print(f"Width in metres: {width_m:.2f}")
    print(f"Height in metres: {height_m:.2f}")
    print()
    
    # All the tiles are exactly 109.8km wide and 109.8km tall. 


minx_m, miny_m, maxx_m, maxy_m





workers = 1
batch_size = math.ceil(len(rows) / workers)
batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
print("num_batches: ", len(batches))
print("num tiles in first batch", len(batches[0]))

# %%time
with ProcessPoolExecutor(max_workers=workers) as executor:
    print(f"Starting {workers} workers, with {batch_size} rows each")
    futures = [executor.submit(run_download, batch) for batch in batches]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Worker failed with: {e}", flush=True)


