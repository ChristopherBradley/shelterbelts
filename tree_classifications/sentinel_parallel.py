# Parallelise the sentinel downloads using a single job instead of hammering the scheduler with lots of jobs from a bash script

import argparse
import os
import sys
import logging
import pickle
import numpy as np
import xarray as xr
import rioxarray
import datacube
from dea_tools.temporal import xr_phenology, temporal_statistics
from dea_tools.datahandling import load_ard
from dea_tools.bandindices import calculate_indices
from dea_tools.plotting import display_map, rgb
from dea_tools.dask import create_local_dask_cluster
import hdstats

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box
import psutil

import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import subprocess



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

def sentinel_download(tif, year, indir, outdir):
    
    tif_id = '_'.join(tif.split('_')[:2])
    print(f"Starting sentinel_download: {tif_id}, {year}\n")

    # Specify the parameters
    filename = os.path.join(indir, tif)
    with rasterio.open(filename) as src:
        bounds = src.bounds
        src_crs = src.crs.to_string()
    lat_range = (bounds[1], bounds[3])
    lon_range = (bounds[0], bounds[2])
    time_range = (f"{year}-01-01", f"{year}-12-31")
    input_crs=src_crs 
    output_crs=src_crs
    query = define_query_range(lat_range, lon_range, time_range, input_crs, output_crs)

    # Load the data
    # client = create_local_dask_cluster(return_client=True)
    dc = datacube.Datacube(app=tif_id)
    ds = load_and_process_data(dc, query)

    # Save the data
    filename = os.path.join(outdir, f'{tif_id}_ds2_{year}.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {filename}")

def run_download(row):
    tif, year = row
    sentinel_download(tif, year, indir, outdir)

# +
# %%time
if __name__ == '__main__':
    indir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
    outdir = '/scratch/xe2/cb8590/Nick_sentinel'

    csv_filename = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
    df_years = pd.read_csv(csv_filename, index_col='filename')
    df_years_2017_2022 = df_years[df_years['year'] >= 2017]

    t1 = "g1_1752_binary_tree_cover_10m.tiff"
    t2 = "g1_2193_binary_tree_cover_10m.tiff"
    t3 = "g1_71513_binary_tree_cover_10m.tiff"
    t4 = "g1_8184_binary_tree_cover_10m.tiff"
    df = df_years_2017_2022.loc[[t1, t2, t3, t4]]

    # df = df_years_2017_2022.sample(n=1, random_state=0)
    rows = list(df[['year']].itertuples(name=None))
    
    with ProcessPoolExecutor(max_workers=len(rows)) as executor:
        print(f"Starting {len(rows)} workers")
        executor.map(run_download, rows)
        
# Took 1 min 13 secs with 5 workers and 5 datacubes. 
# Timed out with 5 workers and 1 datacube.
# -


# csv_filename = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
# df_years = pd.read_csv(csv_filename, index_col='filename')
# df_years_2017_2022 = df_years[df_years['year'] >= 2017]
# df = df_years_2017_2022
# # Files that gave that bad tiff error
# # ERROR 1: TIFFFillTile:Read error at row 256, col 256, tile 17; got 0 bytes, expected 95730
# t1 = "g1_1752_binary_tree_cover_10m.tiff"
# t2 = "g1_2193_binary_tree_cover_10m.tiff"
# t3 = "g1_71513_binary_tree_cover_10m.tiff"
# t4 = "g1_8184_binary_tree_cover_10m.tiff"
# df.loc[[t1, t2, t3, t4]]

