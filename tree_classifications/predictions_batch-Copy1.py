# Make a prediction of tree cover on each tile using the trained model

# +
import os, sys

import glob
import pickle
import ast
import traceback

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import transform

import pyproj
from pyproj import Transformer

import xarray as xr
import rioxarray as rxr
import rasterio

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
import joblib

# ProcessPoolExecutor does not play nicely with tensorflow. Need to do this parallelisation another way.
from concurrent.futures import ProcessPoolExecutor, as_completed
# -

# Change directory to this repo
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
# %%time
# Load the trained model and standard scaler
filename_model = '/g/data/xe2/cb8590/models/nn_89a_92s_85r_86p.keras'
filename_scaler = '/g/data/xe2/cb8590/models/scaler_89a_92s_85r_86p.pkl'

model = keras.models.load_model(filename_model)
scaler = joblib.load(filename_scaler)
# -

# Prepare the colour scheme for the tiff files
cmap = {
    0: (240, 240, 240), # Non-trees are white
    1: (0, 100, 0),   # Trees are green
}


def tif_prediction(tile):
    
    # Load the sentinel imagery
    with open(tile, 'rb') as file:
        ds = pickle.load(file)

    # Calculate vegetation indices
    B8 = ds['nbart_nir_1']
    B4 = ds['nbart_red']
    B3 = ds['nbart_green']
    B2 = ds['nbart_blue']
    ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
    ds['NDVI'] = (B8 - B4) / (B8 + B4)
    ds['GRNDVI'] = (B8 - B3 + B4) / (B8 + B3 + B4)

    print("Preprocessing")

    # Preprocess the temporally and spatially aggregated metrics
    ds_agg = aggregated_metrics(ds)
    variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
    ds_selected = ds[variables] 
    ds_stacked = ds_selected.to_array().transpose('variable', 'y', 'x').stack(z=('y', 'x'))

    # Normalise the inputs using the same standard scaler during training
    X_all = ds_stacked.transpose('z', 'variable').values  # shape: (n_pixels, n_features)
    df_X_all = pd.DataFrame(X_all, columns=ds_selected.data_vars) # Just doing this to silence the warning about not having feature names
    X_all_scaled = scaler.transform(df_X_all)

    # Make predictions and add to the xarray
    # Need to load the model from within the worker or else it gets stuck
    print("Loading model")
    model = keras.models.load_model(filename_model)

    print("Predicting")
    preds = model.predict(X_all_scaled)
    
    print("Predictions done")

    predicted_class = np.argmax(preds, axis=1)
    pred_map = xr.DataArray(predicted_class.reshape(ds.dims['y'], ds.dims['x']),
                            coords={'y': ds.y, 'x': ds.x},
                            dims=['y', 'x'])

    print("About to save the tif")
    
    # Save the predictions as a tif file
    da = pred_map.astype('uint8')
    tile_id = "_".join(tile.split('/')[-1].split('_')[:2])
    filename = f'/scratch/xe2/cb8590/Nick_predicted/{tile_id}_predicted.tif'
    
    # print("Importing rasterio")
    
    print("Using rasterio.open")
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
        # tiled=True,       # Can't be tiled if you want to be able to visualise it in preview. And no point in tiling such a small tif file
        # blockxsize=2**10,
        # blockysize=2**10,
        photometric="palette",
    ) as dst:
        dst.write(da.values, 1)
        dst.write_colormap(1, cmap)
    print("Saved", filename)



def worker_predictions(tiles):
    """Run tile_csv and report more info on errors that occur"""
    for tile in tiles:
        tile_id = "_".join(tile.split('/')[-1].split('_')[:2])
        print(f"Starting tile: {tile_id}", flush=True)
        try:
            tif_prediction(tile)
        except Exception as e:
            print(f"Error in tile {tile_id}:", flush=True)
            traceback.print_exc(file=sys.stdout)



# +
# # %%time
### Benchmarking
# for tile in tiles[:10]:
#     tif_prediction(tile)
    
# 4 secs for 1 tiles, and 40 secs for 10 tiles
# That means that it would take 8 hours to do all in series
# or 10 minutes if I can get 50 workers working at once
# -

rows = tiles[:16]
workers = 4
batch_size = math.ceil(len(rows) / workers)
batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
print("num_batches: ", len(batches))
print("num tiles in first batch", len(batches[0]))

# Save the tiles in each batch in a csv file so I can launch a subprocess that works on each batch
for i, batch in enumerate(batches):
    batch_file = f"/g/data/xe2/cb8590/models/batches/batch_{i}.csv"
    df = pd.DataFrame(batch)
    df.to_csv(batch_file, index=False)
    print("Saved", batch_file)

# %%time
worker_predictions(rows)

# %%time
with ProcessPoolExecutor(max_workers=len(batches)) as executor:
    print(f"Starting {len(batches)} workers, with {batch_size} rows each")
    futures = [executor.submit(worker_predictions, batch) for batch in batches]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Worker failed with: {e}", flush=True)


