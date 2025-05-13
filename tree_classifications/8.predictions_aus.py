import glob
import math
import pandas as pd
import geopandas as gpd
import subprocess
from shapely.geometry import box

import psutil
import gc
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# Change directory to this repo
import sys, os
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


# +
# %%time
from tree_classifications.sentinel_parallel import sentinel_download
from tree_classifications.predictions_batch import tif_prediction_ds

# 1min 38 secs to import predictions_batch... I wonder if I have unnecessary imports I can remove to reduce this?
# -

outdir = '/scratch/xe2/cb8590/ka08_trees'

# Load the sentinel imagery tiles
filename_sentinel_bboxs = "/g/data/xe2/cb8590/Nick_outlines/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
gdf = gpd.read_file(filename_sentinel_bboxs)

test_tile_id = "55HFC"
test_tile = gdf.loc[gdf['Name'] == test_tile_id, 'geometry'].values[0]


polygon = test_tile

# Get centroid of the polygon (in degrees)
centroid = polygon.centroid
lon, lat = centroid.x, centroid.y



if 'ds' in locals():
    del ds

# +
# %%time
# Create a small area in the center for testing
half_deg = 0.02  # 0.01° / 2
bbox = box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)

stub = "test"
outdir = "/scratch/xe2/cb8590/tmp"
year="2020"
bounds = bbox.bounds
src_crs = "EPSG:4326"
ds = sentinel_download(stub, year, outdir, bounds, src_crs)

# Should get a start and finish memory to see how much additional it used
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")  

# +
# sentinel_download with Large compute (7 cores, 32GB)
# 1km x 1km x 1 year = 30, 10, 20, 10, 10, 20s 5GB, 26s 500MB
# 2km x 2km x 1 year = 10, 10, 10, 23s 5GB 
# 4kmx4km = 20s 1.5GB
# 10kmx10km = 34, 25s 5GB, 15s 5GB, 16s 5GB 
# 20km x 20km x 1 year = 69 kernel died, 29s 12GB, 29s 20GB, 30s 20GB, 28s 20GB (but there seems to be 10GB not being used)

# +
# del ds
# gc.collect()
# print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# +
# %%time
# How much time & memory does the sentinel download use compared to the machine learning predictions?
da = tif_prediction_ds(ds, "Test", outdir="/scratch/xe2/cb8590/tmp/", savetif=False)

print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB") 

# +
# Seems like the limiting factor is the sentinel download, not the other preprocessing or machine learning predictions
# 1km x 1km = 3s
# 2km x 2km = 49s, 50s
# 4km x 4km = 60 secs
# 10km x 10km = 
# 20km x 20km = Got stuck > 10 mins
