# +
"""
Description:
This script just downloads SENTINEL2 data (using DEA) as an xarray dataset and saves it as a pickle. Only variables necessary for the 05_shelter.py are included.

Usage:
The script is designed to be executed from the command line, where the user can specify the stub name for file naming and the directory for output files, and other parameters:

Requirements:
- Runs using a python environment designed for DEA Sandbox use on NCI. 
module use /g/data/v10/public/modules/modulefiles
module load dea/20231204
- A fair amount of memory for downloading large regions of data. 

Inputs:
- stub name
- coordinates
- buffer (degrees)
- start/end date

Outputs:
- A pickle dump of the xarray (ds2) containing 4 bands of sentinel data (RGB & NIR), and metadata, for the ROI and TOI
- a pickle containg dict of the query used to generate ds2
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
        description="""Download and save Sentinel data, prepare input image for SAMGeo.
        
Example usage:
python3 Code/00_sentinel.py --stub test --outdir /g/data/xe2/cb8590/Data/shelter --lat -34.3890 --lon 148.4695 --buffer 0.01 --start_time '2020-01-01' --end_time '2020-03-31'""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stub", type=str, required=True, help="Stub name for file naming")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for saved files")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the center of the area of interest")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the center of the area of interest")
    parser.add_argument("--buffer", type=float, required=True, help="Buffer in degrees to define the area around the center point")
    parser.add_argument("--start_time", type=str, required=True, help="Start time for the data query (YYYY-MM-DD)")
    parser.add_argument("--end_time", type=str, required=True, help="End time for the data query (YYYY-MM-DD)")
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
    query = define_query(args.lat, args.lon, args.buffer, (args.start_time, args.end_time))
    ds = load_and_process_data(dc, query)

    # save ds for later
    with open(os.path.join(args.outdir, args.stub + '_ds2.pkl'), 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Data saved successfully to {args.outdir}")

    # save query for record keeping
    with open(os.path.join(args.outdir, args.stub + '_ds2_query.pkl'), 'wb') as f:
        pickle.dump(query, f)

if __name__ == "__main__":
    args = parse_arguments()
    # args = argparse.Namespace(
    #     lat=-42.39062467274229,
    #     lon=147.47938065700737,
    #     buffer=2.5,
    #     stub="Test",
    #     start_time="2019-01-01",
    #     end_time="2019-03-01",
    #     outdir="/g/data/xe2/cb8590/shelterbelts/tasmania_testdata",
    # )
    main(args)


# !ls /g/data/xe2/cb8590




# Load the tiff bboxs and filename dates. Filter to just 2019 for now.
filename_bboxs = '/g/data/xe2/cb8590/Nick_outlines/tiff_footprints.geojson'
filename_years = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
filename_centroids = '/g/data/xe2/cb8590/Nick_outlines/tiff_centroids.geojson'

# Merge the bounding box and year
# %%time
gdf_bboxs = gpd.read_file(filename_bboxs)
df_years = pd.read_csv(filename_years)
gdf_bbox_year = pd.merge(gdf_bboxs, df_years)
filename_bbox_year = '/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg'
gdf_bbox_year.to_file(filename_bbox_year, layer="geometries")
print("Saved", filename_bbox_year)

# Save the centroids for viewing in QGIS coloured by year
gdf_centroids = gpd.read_file(filename_centroids)
gdf_centroid_year = pd.merge(gdf_centroids, df_years)
filename_centroid_year = '/g/data/xe2/cb8590/Nick_outlines/tiff_centroids_years.gpkg'
gdf_centroid_year.to_file(filename_centroid_year, layer="geometries")
print("Saved", filename_centroid_year)

# Double checking the bounds line up with the tiff exactly when I use the right crs. It does. 
tif = df_years.loc[0]['filename']
filename = os.path.join("/g/data/xe2/cb8590/Nick_Aus_treecover_10m", tif)
with rasterio.open(filename) as src:
    bounds = src.bounds
    src_crs = src.crs
geom = box(*bounds)
gdf = gpd.GeoDataFrame([{"geometry": geom}], crs=src_crs)
filename = "/scratch/xe2/cb8590/bounds_g1_01001_binary_tree_cover_10m.gpkg"
gdf.to_file(filename, layer="bounds", driver="GPKG")
print("Saved", filename)

df_years = pd.read_csv(filename_years, index_col='filename')

client = create_local_dask_cluster(return_client=True)
dc = datacube.Datacube(app='Shelter')

# +
# %%time
# Example Sentinel download using the same crs as the tif Nick provided
tif = 'g1_05293_binary_tree_cover_10m.tiff'
year = df_years.loc[tif]['year']
filename = os.path.join("/g/data/xe2/cb8590/Nick_Aus_treecover_10m", tif)
with rasterio.open(filename) as src:
    bounds = src.bounds
    src_crs = src.crs
query = define_query_range((bounds[1], bounds[3]), (bounds[0], bounds[2]), (f"{year}-01-01", f"{year}-12-31"), src_crs, src_crs)

ds = load_and_process_data(dc, query)
filename = "/scratch/xe2/cb8590/g1_05293_red_2019-01-02.tif"
ds.isel(time=0)['nbart_red'].rio.to_raster(filename)
print(filename)

# Took 1 min to load 62 timesteps of a 1kmx1km area

