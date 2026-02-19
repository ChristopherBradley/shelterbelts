# Parallelise the sentinel downloads using a single job instead of hammering the scheduler with lots of jobs from a bash script

# +
import argparse
import os
import sys

import pickle
import rioxarray as rxr
import datacube
import geopandas as gpd


# I should get rid of all the imports that don't get used (basically everything except load_ard)
from dea_tools.datahandling import load_ard
# from dea_tools.plotting import display_map, rgb
# from dea_tools.dask import create_local_dask_cluster  # Not creating a dask cluster, because I'm doing parallelisation using lots of tiles

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


def download_ds2_bbox(bbox, start_date="2020-01-01", end_date="2021-01-01", outdir=".", stub="TEST", save=True, input_crs='epsg:4326'):
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
    # print('lat_range', lat_range)
    # print('lon_range', lon_range)
    # print('time_range', time_range)
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
    
    return ds

def run_download_gdf(gdf, outdir, start_date='2020-01-01', end_date='2021-01-01'):
    """Download sentinel imagery for each bbox year in the gdf.
        gdf should have columns: filename, year, geometry (with the geom being just the bounding box)
    """
    for i, row in gdf.iterrows():
        filename = row['filename']  # Using the filename as the stub
        if 'start_date' in gdf.columns and 'end_date' in gdf.columns:
            start_date = row['start_date']  # Preference the start date and end dates in the gdf if they exist
            end_date = row['end_date']      # The old method only allowed specifying a full year of data: start_date = f'{year}-01-01'
        stub = filename.split('.')[0] + '_' + start_date[:4] # I need to add the year to the stub if I want to download multiple years per geometry
        bounds = row['geometry'].bounds
        
        # crs = row['crs'] if 'crs' in gdf.columns else gdf.crs. Assuming epsg:4326 since that's what the sentinel download needs to work properly
        
        try:
            print(f"Downloading: {stub}_{start_date}_{end_date}", flush=True)
            download_ds2_bbox(bounds, start_date, end_date, outdir, stub)
        except Exception as e:
            print(f"Error in downloading: {stub}_{start_date}_{end_date}:", flush=True)
            traceback.print_exc(file=sys.stdout)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("gpkg", type=str, help="A gpkg with: (filename, [start_date], [end_date], geometry), generated by util/bounding_boxes.py. If the start_date and end_date columns exist, then they override the extra command line arguments.")
    parser.add_argument("outdir", type=str, help="Output directory for the pickle files")
    parser.add_argument("--limit", type=int, default=None, help="Number of files to process")
    parser.add_argument("--start_date", type=str, default='2020-01-01', help="Starting date to download sentinel imagery (default='2020-01-01')")
    parser.add_argument("--end_date", type=str, default='2021-01-01', help="Ending date to download sentinel imagery (default='2021-01-01')")

    return parser.parse_args()


# Before running this, I downloaded laz files from ELVIS, converted to tifs with lidar.py, and copied from my computer to gadi with rsync
if __name__ == '__main__':
    
    args = parse_arguments()
    
    gdf = gpd.read_file(args.gpkg)

    if args.limit is not None:
        gdf = gdf[:args.limit]
    
    run_download_gdf(gdf, args.outdir, args.start_date, args.end_date)

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
# +
# folder = '/scratch/xe2/tmp/tif_folder'
# gdf = gpd.read_file('/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_nsw.gpkg')
# bbox = gdf.iloc[0]['geometry'].bounds
# bbox2 = gdf.iloc[:1].to_crs('EPSG:3857').iloc[0]['geometry'].bounds
# ds = download_ds2_bbox(bbox2, input_crs='epsg:3857')
# ds
# -


