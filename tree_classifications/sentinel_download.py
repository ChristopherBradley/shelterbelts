# +
"""
Description:
This script downloads Sentinel 2 data for each tiff bbox year in the csv.

Requirements:
- Runs using a python environment designed for DEA Sandbox use on NCI. 
module use /g/data/v10/public/modules/modulefiles
module load dea/20231204
- A fair amount of memory for downloading large regions of data. 

Inputs:
- Input folder with tiff files
- Input csv filename (columns are filename, year)
- Output folder for Sentinel Pickles

"""
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

# -

# Adjust logging configuration for the script
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Download and save Sentinel data
        
Example usage:
python3 sentinel_download.py --indir /g/data/xe2/cb8590/Nick_Aus_treecover_10m  --csv_filename '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv' --outdir /scratch/xe2/cb8590/Nick_sentinel""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--indir", type=str, required=True, help="Directory containing tree binary classification tiff files")
    parser.add_argument("--csv_filename", type=str, required=True, help="CSV containing filenames we want to download and the corresponding year")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for saved files")
    return parser.parse_args()

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


# Single point with km buffer
def define_query_km(lat, lon, buffer, time_range):

    # Specifying a buffer in km instead of degrees, to get square tiles instead of rectangles
    import pyproj
    buffer_m = buffer * 1000
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:7855", always_xy=True).transform
    unproject = pyproj.Transformer.from_crs("EPSG:7855", "EPSG:4326", always_xy=True).transform
    x, y = project(lon, lat)
    min_coord = unproject(x - buffer_m, y - buffer_m)
    max_coord = unproject(x + buffer_m, y + buffer_m)
    lon_range = (min_coord[0], max_coord[0])
    lat_range = (min_coord[1], max_coord[1])

    # lat_range = (lat-buffer, lat+buffer)
    # lon_range = (lon-buffer, lon+buffer)
    query = {
        'centre': (lat, lon),
        'y': lat_range,
        'x': lon_range,
        'time': time_range,
        'resolution': (-10, 10),
        'output_crs': 'epsg:6933',
        'group_by': 'solar_day'
    }
    # note that centre is not recognized as query option in load_arc, but we want to output it as a record.
    return query


# Single point with degrees buffer (the version John created)
def define_query(lat, lon, buffer, time_range):

    lat_range = (lat-buffer, lat+buffer)
    lon_range = (lon-buffer, lon+buffer)
    query = {
        'centre': (lat, lon),
        'y': lat_range,
        'x': lon_range,
        'time': time_range,
        'resolution': (-10, 10),
        'output_crs': 'epsg:6933', # Should use EPSG:7844 when working with imagery in Australia
        'group_by': 'solar_day'
    }
    # note that centre is not recognized as query option in load_arc, but we want to output it as a record.
    return query

def load_and_process_data(dc, query):
    query = {k: v for k, v in query.items() if k != 'centre'} # this takes centre out of the query	
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


def main(args):
    client = create_local_dask_cluster(return_client=True)
    dc = datacube.Datacube(app='Shelter')
    
    indir = args.indir
    csv_filename = args.csv_filename
    outdir = args.outdir
    print("Starting sentinel_download:", indir, csv_filename, outdir)
    
    df_years = pd.read_csv(csv_filename, index_col='filename')
    for tif in df_years.index:
        year = df_years.loc[tif]['year']
        filename = os.path.join("/g/data/xe2/cb8590/Nick_Aus_treecover_10m", tif)
        with rasterio.open(filename) as src:
            bounds = src.bounds
            src_crs = src.crs.to_string()
        lat_range = (bounds[1], bounds[3])
        lon_range = (bounds[0], bounds[2])
        time_range = (f"{year}-01-01", f"{year}-12-31")
        input_crs=src_crs 
        output_crs=src_crs
        query = define_query_range(lat_range, lon_range, time_range, input_crs, output_crs)

        ds = load_and_process_data(dc, query)

        # save ds for later
        filename = os.path.join(args.outdir, f'tif_ds2_{year}.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {filename}")


# %%time
if __name__ == "__main__":
    # args = parse_arguments()
    args = argparse.Namespace(
        indir='/g/data/xe2/cb8590/Nick_Aus_treecover_10m',
        csv_filename='/g/data/xe2/cb8590/Nick_outlines/gdf_x5.csv',
        outdir='/scratch/xe2/cb8590/Nick_sentinel'
    )
    main(args)


# +
# # Prepping files for gradually scaling up on gadi
# csv_filename = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
# df_years = pd.read_csv(csv_filename, index_col='filename')
# df_years_2017_2022 = df_years[df_years['year'] >= 2017]

# filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_01001.csv'
# df_years_2017_2022.iloc[0:1].to_csv(filename)

# filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_x5.csv'
# df_years_2017_2022.iloc[0:5].to_csv(filename)

# filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_x100.csv'
# df_years_2017_2022.sample(n=100, random_state=0).to_csv(filename)

# filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_2017_2022.csv'
# df_years_2017_2022.to_csv(filename)
