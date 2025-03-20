# +
# Create 5km woody veg tiles to match the sentinel training data
# -

import rasterio
import pandas as pd
import numpy as np
from rasterio.windows import Window
from rasterio.transform import rowcol
import pyproj
import xarray as xr
import pickle
import rioxarray as rxr
from rasterio.enums import Resampling

# Change directory to the repo root
import os
import sys
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/shelterbelts")
elif os.path.basename(os.getcwd()) != "shelterbelts":
    paddockTS_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from PaddockTS root
    paddockTS_dir = os.getcwd()
os.chdir(paddockTS_dir)
sys.path.append(paddockTS_dir)
print(paddockTS_dir)

# Double checking the satellite imagery xarrays line up with the tiles. They almost do (EPSG:4326 vs EPSG:6933)
filename = "data/41_80_147_34_ds2.pkl"
with open(filename, 'rb') as file:
    ds2 = pickle.load(file)
filename = 'ds_41_80_147_34_ds2_red.tif'
ds2['nbart_red'].rio.to_raster(filename)

# Load the woody veg tiff file
filename = "data/Tas_WoodyVeg_201903_v2.2.tif"

# Load the tile coordinates
filename = "data/lidar_tiles_tasmania.csv"

# +
# For each tile coordinate, slice the woody veg and save as a smaller tiff with a matching filename to the sentinel
# -

raster_path = "data/Tas_WoodyVeg_201903_v2.2.tif"
csv_path = "data/lidar_tiles_tasmania.csv"
output_dir = "data/woody_veg_tiles"

df = pd.read_csv(csv_path)

ds = rxr.open_rasterio(raster_path)

# %%time
# Make the woody veg EPSG match the satellite imagery imagery. 
# The alternative it to specify EPSG:3577 when downloading the satellite imagery from DEA
ds = ds.rio.reproject("EPSG:6933", resampling=Resampling.bilinear)
# Took 2 mins 30 secs

# %%time
ds_matched = ds.rio.reproject_match(ds2)
ds_matched.rio.to_raster(output_file)
print(output_file)
