# +
import glob
import math
import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import rotate

from tensorflow import keras
import subprocess
import joblib
# -

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
from tree_classifications.merge_inputs_outputs import aggregated_metrics

# 1min 38 secs to import predictions_batch... I wonder if I have unnecessary imports I can remove to reduce this?
# -
outdir = '/scratch/xe2/cb8590/ka08_trees'

# %%time
# Load the sentinel imagery tiles
filename_sentinel_bboxs = "/g/data/xe2/cb8590/Nick_outlines/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
gdf = gpd.read_file(filename_sentinel_bboxs)

gdf.crs

# +
# (Testing parallelisation issues)
# Choose 4 test tiles directly adjacent
# label them top left, top right, bottom left, bottom right
# Choose the 20km region at the adjacent boundary of each tile
# Run the 4 tiles in parallel to see if I get any errors

# +
# Testing tiles that should not give any parallelisation errors
# Choose 4 tiles not touching each other
# Split each one into 20kmx20km sections
# Start 4 workers, each one with 25 subtiles
# -



test_tiles_unconnected = "55HFC", "55HDC", "55HDA", "55HFA"
test_tile_id = "55HFC"
test_tile = gdf.loc[gdf['Name'] == test_tile_id, 'geometry'].values[0]


polygon = test_tile

# Get centroid of the polygon (in degrees)
centroid = polygon.centroid
lon, lat = centroid.x, centroid.y
half_deg = 0.1 
bbox = box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)

# +
# Create 25 equally spaced tiles within the larger tile
gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

# 1. Automatically determine appropriate UTM zone
utm_crs = gdf.estimate_utm_crs()
gdf_utm = gdf.to_crs(utm_crs)
polygon_utm = gdf_utm.geometry.iloc[0]

# 2. Compute minimum rotated rectangle and its angle
min_rect = polygon_utm.minimum_rotated_rectangle
coords = list(min_rect.exterior.coords)
dx = coords[1][0] - coords[0][0]
dy = coords[1][1] - coords[0][1]
angle = math.degrees(math.atan2(dy, dx))

# 3. Rotate the polygon to axis-align it
origin = polygon_utm.centroid
rotated_polygon = rotate(polygon_utm, -angle, origin=origin)

# 4. Create grid inside the bounds of rotated polygon
minx, miny, maxx, maxy = rotated_polygon.bounds
grid_size = 5
dx = (maxx - minx) / grid_size
dy = (maxy - miny) / grid_size

tiles = []
names = []

for i in range(grid_size):
    for j in range(grid_size):
        x0 = minx + j * dx
        y0 = miny + i * dy
        x1 = x0 + dx
        y1 = y0 + dy
        tile = box(x0, y0, x1, y1)

        # Rotate tile back to original orientation
        tile_rotated = rotate(tile, angle, origin=origin)

        tiles.append(tile_rotated)

        # Create a name from centroid
        lon, lat = gpd.GeoSeries([tile_rotated], crs=utm_crs).to_crs("EPSG:4326").centroid.iloc[0].xy
        name = f"{test_tile_id}{np.round(lat[0], 2)}_{np.round(lon[0], 2)}".replace(".", "_")
        names.append(name)

# 5. Create GeoDataFrame and reproject back to EPSG:4326
tiles_gdf = gpd.GeoDataFrame({'name': names, 'geometry': tiles}, crs=utm_crs)
tiles_wgs84 = tiles_gdf.to_crs("EPSG:4326")

# 6. Save to GeoPackage
filename = "/scratch/xe2/cb8590/tmp/55HFC_grid_tiles.gpkg"
tiles_wgs84.to_file(filename, layer="tiles", driver="GPKG")
# -

tiles_wgs84

# +
# %%time
# Create a small area in the center for testing

stub = "test"
outdir = "/scratch/xe2/cb8590/tmp"
year="2020"
bounds = bbox.bounds
src_crs = "EPSG:4326"

ds = sentinel_download(stub, year, outdir, bounds, src_crs)
da = tif_prediction_ds(ds, "Test", outdir="/scratch/xe2/cb8590/tmp/", savetif=False)

# -



# +
# sentinel_download with Large compute (7 cores, 32GB)
# 1km x 1km x 1 year = 30, 10, 20, 10, 10, 20s 5GB, 26s 500MB
# 2km x 2km x 1 year = 10, 10, 10, 23s 5GB 
# 4kmx4km = 20s 1.5GB, 20s 1.5GB
# 10kmx10km = 34, 25s 5GB, 15s 5GB, 16s 5GB, 28s 5GB
# 20km x 20km x 1 year = 69 kernel died, 29s 12GB, 29s 20GB, 30s 20GB, 28s 20GB (but there seems to be 10GB not being used)

# Preprocessing + neural network predictions 
# (the predictions are the slowest part, presumably faster on GPU)
# 4km: 20 secs
# 10km: 1 min 17 secs
# 20km: 5 mins as predicted.

# Scaling up estimations
# If it takes 5 mins for 20kmx20km, that's 25*5 = 125 mins = 2 hours per Sentinel Tile. 
# There are about 30x30 = 900 Sentinel tiles in Australia
# Hopefully I can do tiles in parallel, so that will be about 20 hours on one Node, 
# Or just 2 hours if it scales nicely up to 20 nodes (1000 CPUS)
