# Parallelise the sentinel downloads using a single job instead of hammering the scheduler with lots of jobs from a bash script

# +
import argparse
import os
import sys
import psutil
import ast

import csv
import pickle
import numpy as np
import xarray as xr
import rioxarray
import datacube
import hdstats
import pandas as pd
import geopandas as gpd
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

def sentinel_download(tif, year, outdir, bounds, src_crs):
    
    tif_id = '_'.join(tif.split('_')[:2])
    # print(f"Starting sentinel_download: {tif_id}, {year}\n")

    # Prep the DEA query
    lat_range = (bounds[1], bounds[3])
    lon_range = (bounds[0], bounds[2])
    time_range = (f"{year}-01-01", f"{year}-12-31")
    input_crs=src_crs 
    output_crs=src_crs
    query = define_query_range(lat_range, lon_range, time_range, input_crs, output_crs)

    # Load the data
    dc = datacube.Datacube(app=tif_id)
    ds = load_and_process_data(dc, query)

    # Save the data
    filename = os.path.join(outdir, f'{tif_id}_ds2_{year}.pkl')

    with open(filename, 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {filename}", flush=True)

def run_download(row):
    try:
        tif, year, bounds, crs = row
        bbox = ast.literal_eval(bounds) # The bbox is saved as a string initially because I stored it in a csv file with pandas
        print(f"Worker running: {tif}_{year}", flush=True)
        sentinel_download(tif, year, outdir, bbox, crs)
    except Exception as e:
        print(f"Error in worker {tif}_{year}:", flush=True)
        traceback.print_exc(file=sys.stdout)
        raise

# +
# # %%time
# if __name__ == '__main__':

# indir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
# outdir = '/scratch/xe2/cb8590/Nick_sentinel'

# csv_filename = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
# df_years = pd.read_csv(csv_filename, index_col='filename')
# df_years_2017_2022 = df_years[df_years['year'] >= 2017]

# df = df_years_2017_2022.sample(n=1000, random_state=0)
# rows = list(df[['year']].itertuples(name=None))

# # Pre-load the crs and bounds for each tif file
# rows2 = []
# import rasterio
# for row in rows:
#     tif, year = row
#     filename = os.path.join(indir, tif)
#     with rasterio.open(filename) as src:
#         bounds = src.bounds
#         crs = src.crs.to_string()
#     rows2.append((tif, year, bounds, crs))

# # I think the TIFF Read Error gets hidden in a jupyter notebook, so you only see it when doing qsub. But some workers seem to fail silently right now and don't download the pickle file.
# with ProcessPoolExecutor(max_workers=len(rows)) as executor:
#     print(f"Starting {len(rows)} workers")
#     executor.map(run_download, rows2)
# -

# Find the overlapping sentinel tiles for each tree cover tif
def find_overlapping_tiles(row, sentinel_gdf):
    bbox_string = row['bbox']
    bbox = ast.literal_eval(bbox_string)

    src_crs = CRS.from_user_input(row['crs'])
    dst_crs = sentinel_gdf.crs

    # Create shapely box and reproject
    geom = box(*bbox)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x1, y1 = transformer.transform(bbox[0], bbox[1])
    x2, y2 = transformer.transform(bbox[2], bbox[3])
    reprojected_geom = box(x1, y1, x2, y2)

    # Find overlapping tiles
    overlaps = sentinel_gdf[sentinel_gdf.intersects(reprojected_geom)]
    return overlaps['Name'].tolist()


# Create groups that don't have overlapping tile_years to avoid concurrent tiff access errors
def group_rows_by_conflict(rows2_df):
    df = rows2_df.copy().reset_index(drop=True)
    G = nx.Graph()

    # Add a node for each row
    for idx in df.index:
        G.add_node(idx)

    # Map key → list of row indices that share it
    key_to_rows = defaultdict(list)
    for idx, keys in df['year_tile_keys'].items():
        for key in keys:
            key_to_rows[key].append(idx)

    # Add edges between all rows that share a key
    for row_idxs in key_to_rows.values():
        for i, j in combinations(row_idxs, 2):
            G.add_edge(i, j)

    # Color the graph (each color = group)
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')

    # Assign group numbers
    df['group'] = df.index.map(coloring)
    return df


indir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
outdir = '/scratch/xe2/cb8590/Nick_sentinel'
filename_bbox_year = "/g/data/xe2/cb8590/Nick_outlines/nick_bbox_year_crs.csv"
filename_gdf_maxyear = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"


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


# +
# Just trying out the gs_3117 tiff which failed in the qsub. It worked fine, confirming again this is a parallelisation error. 
# rows = df[df['tif'] == 'g2_3117_binary_tree_cover_10m.tiff'][['tif', 'year', 'bbox', 'crs']].values.tolist()
# -

# Load the tiff bboxs
df = pd.read_csv(filename_bbox_year)
df_2017_2022 = df[df['year'] >= 2017]

# %%time
# Load the sentinel tile bboxs, downloaded from: https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index
filename_sentinel_bboxs = "/g/data/xe2/cb8590/Nick_outlines/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
gdf_sentinel_bboxs = gpd.read_file(filename_sentinel_bboxs)

# +
# %%time
# Assigning a sentinel tile and year to each row to try to avoid concurrent access of the same location
df['sentinel_tiles'] = df_2017_2022.apply(find_overlapping_tiles, axis=1, sentinel_gdf=gdf_sentinel_bboxs)

# Replace NaN with []
df['sentinel_tiles'] = df['sentinel_tiles'].apply(lambda x: [] if isinstance(x, float) and np.isnan(x) else x)
# -

# Create a key with the year_tile combinations
df['year_tile_keys'] = df.apply(
    lambda row: [f"{row['year']}_{tile}" for tile in row['sentinel_tiles']],
    axis=1
)

df.sample(100)[:100]

grouped_df = group_rows_by_conflict(df)

grouped_df['group'].value_counts()

df.iloc[0]['bbox']

# After all that funny overlapping grouping stuff, I'm wondering if it might just be easiest to run in batches of 50 anyway
rows = df[['tif', 'year', 'bbox', 'crs']].values.tolist()
batch_size = 50
batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

batches = batches[20:30]

# +
# %%time
for i, batch in enumerate(batches):
    rows = batch
    with ProcessPoolExecutor(max_workers=len(rows)) as executor:
        print(f"Starting {len(rows)} workers for batch {i}")
        futures = [executor.submit(run_download, row) for row in rows]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with: {e}", flush=True)
                
# 38 mins for 50 workers x 10 batches
# -


