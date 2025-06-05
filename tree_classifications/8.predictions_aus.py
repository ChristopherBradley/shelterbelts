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


# %%time
from tree_classifications.sentinel_parallel import sentinel_download
from tree_classifications.predictions_batch import tif_prediction_ds
from tree_classifications.merge_inputs_outputs import aggregated_metrics

outdir = '/scratch/xe2/cb8590/ka08_trees'
outdir_batches = "/g/data/xe2/cb8590/models/batches_aus"
# outdir_batches = "/g/data/xe2/cb8590/models/batches_aus_10km"

# %%time
# Load the sentinel imagery tiles
filename_sentinel_bboxs = "/g/data/xe2/cb8590/Nick_outlines/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
gdf_sentinel = gpd.read_file(filename_sentinel_bboxs)


# +
def sub_tiles(gdf_sentinel, tile_id, grid_size=5):
    """Create 25 equally spaced tiles within the larger sentinel tile"""

    polygon = gdf_sentinel.loc[gdf_sentinel['Name'] == tile_id, 'geometry'].values[0]

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
    minx, miny, maxx, maxy = rotated_polygon.bounds

    # 4. Create grid inside the bounds of rotated polygon
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
            name = f"{tile_id}{np.round(lat[0], 2)}_{np.round(lon[0], 2)}".replace(".", "_")
            names.append(name)

    # 5. Create GeoDataFrame and reproject back to EPSG:4326
    tiles_gdf = gpd.GeoDataFrame({'stub': names, 'geometry': tiles}, crs=utm_crs)
    tiles_wgs84 = tiles_gdf.to_crs("EPSG:4326")

    # 6. Save to GeoPackage
    # filename = "/scratch/xe2/cb8590/tmp/55HFC_grid_tiles.gpkg"
    filename = os.path.join(outdir_batches, f"{tile_id}.gpkg")
    if os.path.exists(filename):
        os.remove(filename)
    tiles_wgs84.to_file(filename, layer="tiles", driver="GPKG")
    print("Saved", filename)
    
    return tiles_wgs84
    
# gdf = sub_tiles(gdf_sentinel, "55HFC")
# len(gdf)
# -

# I want to try relaunching the subprocess for each for full tile to see if that solves memory accumulation issues.

outdir_batches = "/g/data/xe2/cb8590/models/batches_aus_100km"
test_tiles_unconnected = [["55HDA"], # Number of tiles per row is the number of subprocesses we launch each time
                          ["55HFA"],
                          ["55HFC"],
                          ['55HDC']  # Number of rows is the number of times we relaunch the subprocesses
                        ]   
 # 
# test_tiles_connected = ["55HFC", "55HEC", "55HEB", "55HFB",
#                         "55HGC", "56HKH", "55HGB", "56HKG",
                        # "55HEV", "55HFV", "55HEU", "55HFU",
                        # "55HGV", "56HKE", "55HGU", "56HKD"]

all_tiles = test_tiles_unconnected

for tile_batch in all_tiles:

    for tile_id in tile_batch:
        # Create a geopackage with a list of stubs and bboxs for this subprocess to work on
        sub_tiles(gdf_sentinel, tile_id, grid_size=1)

    # Recreate the filename for each of those geopackages we just saved
    batch_files = [os.path.join(outdir_batches, f"{tile_id}.gpkg") for tile_id in tile_batch]

    # Launch a bunch of subprocesses in parallel
    procs = []
    for batch_file in batch_files:
        p = subprocess.Popen(["python", "tree_classifications/predictions_batch.py", "--csv", batch_file])
        procs.append(p)
        
    # Wait before relaunching more subprocesses
    for p in procs:
        p.wait()

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
