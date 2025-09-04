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
import rioxarray as rxr
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
# from dea_tools.dask import create_local_dask_cluster  # Not creating a dask cluster, because I'm doing parallelisation using lots of tiles
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

# Specific lat and lon range (instead of just lat, lon, buffer) so I can match the tiff files exactly
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
    with contextlib.redirect_stdout(io.StringIO()):     # Silence the print statements
        ds = load_ard(
            dc=dc,
            products=['ga_s2am_ard_3', 'ga_s2bm_ard_3', 'ga_s2cm_ard_3'],
            cloud_mask='s2cloudless',
            min_gooddata=0.9,
            measurements=['nbart_blue', 'nbart_green', 'nbart_red', 
                          'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
                          'nbart_nir_1', 'nbart_nir_2',
                          'nbart_swir_2', 'nbart_swir_3'],
            **query
        )
    return ds

def download_ds2(tif, start_date="2020-01-01", end_date="2021-01-01", outdir=".", save=True):
    """Download sentinel imagery matching the bounding box of the tif file"""
    da = rxr.open_rasterio(tif).isel(band=0).drop_vars('band')
    da_4326 = da.rio.reproject('EPSG:4326')
    bbox = da_4326.rio.bounds()
    stub = tif.split('/')[-1].split('.')[0]
    ds2 = download_ds2_bbox(bbox, start_date, end_date, outdir, stub, save)
    return ds2


# +
def download_ds2_bbox(bbox, start_date="2020-01-01", end_date="2021-01-01", outdir=".", stub="TEST", save=True):
    """
    Download sentinel imagery for the bounding box and time period of interest.

    Parameters:
        bbox: bounding box of the region of interest
        start_date: First date of imagery to download
        end_date: Last date of imagery to download
        outdir: Output folder for the pickle file
        stub: Prefix of the pickle file
        save: Whether to save to file

    Returns:
        Dataset: The loaded xarray Dataset (also saved to `{outdir}/{stub}.pkl`).
    """
    
    # Prep the DEA query
    lat_range = (bbox[1], bbox[3])
    lon_range = (bbox[0], bbox[2])
    time_range = (start_date, end_date)
    print('lat_range', lat_range)
    print('lon_range', lon_range)
    print('time_range', time_range)
    input_crs='epsg:4326' # I'm not sure it actually works with other crs' as input
    output_crs='EPSG:3857'  # It takes about 50% longer with EPSG:28355 or EPSG:3577.
    query = define_query_range(lat_range, lon_range, time_range, input_crs, output_crs)

    # Load the data
    dc = datacube.Datacube(app='sentinel_download')
    ds = load_and_process_data(dc, query)

    # Save the data
    if save:
        year = start_date[:4]
        filename = os.path.join(outdir, f'{stub}_ds2_{year}.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {filename}", flush=True)
    
    
# -

def run_download_gdf(gdf, outdir):
    """Download sentinel imagery for each bbox year in the gdf.
        gdf should have columns: filename, year, geometry (with the geom being just the bounding box)
    """
    for i, row in gdf.iterrows():
        filename = row['filename']
        stub = filename.split('.')[0]
        year = row['year']
        start_date = f'{year}-01-01'
        end_date = f'{year}-01-01'
        crs = row['crs'] if 'crs' in gdf.columns else gdf.crs
        bounds = row['geometry'].bounds
        try:
            print(f"Downloading: {stub}_{year}", flush=True)
            download_ds2_bbox(bbox, start_date, end_date, outdir, stub)
        except Exception as e:
            print(f"Error in downloading: {stub}_{year}:", flush=True)
            traceback.print_exc(file=sys.stdout)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gpkg", type=str, required=True, help="filename of the gpkg with: (filename, year, geometry) for each tif")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for the pickle files")
    parser.add_argument("--limit", type=int, default=None, help="Output directory for the pickle files")

    return parser.parse_args()


# Before running this, I downloaded laz files from ELVIS, converted to tifs with lidar.py, and copied from my computer to gadi with rsync
if __name__ == '__main__':
    
    args = parse_arguments()
    
    gpkg = args.gpkg
    outdir = args.outdir
    limit = args.limit
    
    gdf = gpd.read_file(gpkg)
    
    if limit is not None:
        gdf = gdf[:limit]
    
    run_download_gdf(gdf, outdir)

# +
# gpkg = '/scratch/xe2/cb8590/Tas_tifs/tas_lidar_tif_attributes.gpkg'
# outdir = '/scratch/xe2/cb8590/Tas_sentinel'
# gdf = gpd.read_file(gpkg)
# gdf = gdf[:1]
# run_download_gdf(gdf, outdir)

# +
# tif  = 'Projects/shelterbelts/data/Fulham_worldcover.tif'
# outdir = '/scratch/xe2/cb8590/tmp'
# stub = 'TEST_Fulham'
# download_ds2(tif, start_date="2020-01-01", end_date="2021-01-01", outdir=".")
# # 10 secs using the NCI datacube compared to 5 mins using the DEA STAC API
# -


