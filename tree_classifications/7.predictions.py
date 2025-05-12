# Make a prediction of tree cover on each tile using the trained model

# +
import glob
import pickle
import ast
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import transform

import pyproj
from pyproj import Transformer

import rasterio
import xarray as xr
import rioxarray as rxr

from tensorflow import keras
import joblib

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

# Load the filenames for all the sentinel tiles I've downloaded
sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
tiles = glob.glob(f'{sentinel_dir}/*.pkl')
print(len(tiles))

# +
# Load the trained model and standard scaler
filename = '/g/data/xe2/cb8590/models/nn_89a_92s_85r_86p.keras'
model = keras.models.load_model(filename)

filename_scaler = '/g/data/xe2/cb8590/models/scaler_89a_92s_85r_86p.pkl'
scaler = joblib.load(filename_scaler)
# -

tile = tiles[0]

tile_id = "_".join(tile.split('/')[-1].split('_')[:2])
tile_id

# +
# Load the sentinel imagery
with open(tile, 'rb') as file:
    ds = pickle.load(file)

# I should make the vegetation indices, aggregated metrics & normalisation a function that I can reuse for training and applying the neural network

# -

# Calculate vegetation indices
B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B3 = ds['nbart_green']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
ds['NDVI'] = (B8 - B4) / (B8 + B4)
ds['GRNDVI'] = (B8 - B3 + B4) / (B8 + B3 + B4)

# Preprocess the temporally and spatially aggregated metrics
ds_agg = aggregated_metrics(ds)
variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
ds_selected = ds[variables] 

ds_selected.data_vars

ds_stacked = ds_selected.to_array().transpose('variable', 'y', 'x').stack(z=('y', 'x'))
ds_stacked.shape

# Normalise the inputs using the same standard scaler during training
X_all = ds_stacked.transpose('z', 'variable').values  # shape: (n_pixels, n_features)
X_all_scaled = scaler.transform(X_all)

# Make predictions and add to the xarray
preds = model.predict(X_all_scaled, batch_size=1024)  # shape: (n_pixels, n_classes)
predicted_class = np.argmax(preds, axis=1)
pred_map = xr.DataArray(predicted_class.reshape(ds.dims['y'], ds.dims['x']),
                        coords={'y': ds.y, 'x': ds.x},
                        dims=['y', 'x'])
ds['predictions'] = pred_map

ds['predictions'].astype('uint8')

da = ds['predictions'].astype('uint8') + 1

da

# +
# Save the predictions
filename = f'/scratch/xe2/cb8590/Nick_predicted/{tile_id}_predicted.tif'
da = ds['predictions'].astype('uint8')
cmap = {
    0: (240, 240, 240), # Non-trees are white
    1: (0, 100, 0),   # Trees are green
}
with rasterio.open(
    filename,
    "w",
    driver="GTiff",
    height=da.shape[0],
    width=da.shape[1],
    count=1,
    dtype="uint8",
    crs=da.rio.crs,
    transform=da.rio.transform(),
    compress="LZW",
    # tiled=True,       # Can't be tiled if you want to be able to visualise it in preview
    # blockxsize=2**10,
    # blockysize=2**10,
    photometric="palette",
) as dst:
    dst.write(da.values, 1)
    dst.write_colormap(1, cmap)
    
print(filename)
# -

# !du -sh /scratch/xe2/cb8590/Nick_predicted/g1_05079_predicted.tif


# +
# No point in using gdaladdo on a file that's only 100x100 pixels
# # !gdaladdo {filename} 2 4 8 16 32 64
