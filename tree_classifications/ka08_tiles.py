# Make a prediction of tree cover on each tile using the trained model

# +
import glob
import pickle
import ast
import traceback
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, box
from shapely.ops import transform

import pyproj
from pyproj import Transformer

import rasterio
from rasterio.warp import transform_bounds
import rioxarray as rxr
from tensorflow import keras

from concurrent.futures import ProcessPoolExecutor, as_completed
# -

# Change directory to this repo
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:
    repo_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from repo root
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)
print(f"Running from {repo_dir}")

from tree_classifications.merge_inputs_outputs import aggregated_metrics

filename = '/g/data/ka08/ga/ga_s2bm_ard_3/51/JYG/2021/06/11/20210611T041036/ga_s2bm_nbart_3-2-1_51JYG_2021-06-11_final_band04.tif'

# !ls /g/data/ka08/ga

# !ls /g/data/ka08/ga/ga_s2am_ard_3

# !ls /g/data/ka08/ga/ga_s2bm_ard_3

# !ls /g/data/ka08/ga/ga_s2cm_ard_3

# !ls /g/data/ka08/ga/ga_s2am_ard_3/50

# !ls /g/data/ka08/ga/ga_s2bm_ard_3/49

# !ls /g/data/ka08/ga/ga_s2bm_ard_3/50/JNP/2020/06/12

# +
# So I just want to know the bbox for all the 3 digit acronyms in each of the 18 regions
# -

satellites = 'ga_s2am_ard_3', 'ga_s2bm_ard_3', 'ga_s2cm_ard_3'

# Get the 3 digit acronyms for each of the 18 epsg regions
regions = dict()
for satellite in satellites:
    epsg_regions = glob.glob(f"/g/data/ka08/ga/{satellite}/*")
    epsg_regions = [region.split('/')[-1] for region in epsg_regions]
    for epsg_region in epsg_regions:
        if epsg_region not in regions:
            regions[epsg_region] = set()
        tiles = glob.glob(f"/g/data/ka08/ga/{satellite}/{epsg_region}/*")
        tiles = [tile.split('/')[-1] for tile in tiles]
        for tile in tiles:
            regions[epsg_region].add(tile)
regions.keys()

# Number of tiles in each epsg region
sorted([region, len(regions[region])] for region in regions.keys())

region = '50'
tile = 'JNN'

# +
# %%time
root_path = Path('/g/data/ka08/ga/')
sat_folders = ['ga_s2am_ard_3', 'ga_s2bm_ard_3', 'ga_s2cm_ard_3']
target_crs = "EPSG:4326"

features = []

for sat_folder in sat_folders:
    sat_path = root_path / sat_folder
    for epsg_zone in map(str, range(42, 60)):
        epsg_path = sat_path / epsg_zone
        if not epsg_path.exists():
            continue
        for tile_folder in epsg_path.iterdir():
            if not tile_folder.is_dir() or len(tile_folder.name) != 3:
                continue
            # Find any .tif file inside this tile folder
            tif_files = list(tile_folder.rglob('*.tif'))
            if not tif_files:
                continue
            tif_file = tif_files[0]
            try:
                with rasterio.open(tif_file) as src:
                    bounds = src.bounds
                    src_crs = src.crs
                    # Reproject bounds to target CRS
                    reprojected_bounds = transform_bounds(src_crs, target_crs,
                                                          bounds.left, bounds.bottom,
                                                          bounds.right, bounds.top,
                                                          densify_pts=21)
                    geom = box(*reprojected_bounds)
                    features.append({
                        'tile': tile_folder.name,
                        'satellite': sat_folder,
                        'epsg': epsg_zone,
                        'geometry': geom
                    })
            except Exception as e:
                print(f"Failed to read {tif_file}: {e}")

# Create GeoDataFrame in target CRS
gdf = gpd.GeoDataFrame(features, crs=target_crs)


# -

# Save to GeoPackage
filename_ka08 = '/g/data/xe2/cb8590/models/ka08_tile_bounds.gpkg'
gdf.to_file(filename_ka08, driver='GPKG')


