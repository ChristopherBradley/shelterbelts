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

import gc
import psutil
process = psutil.Process(os.getpid())


# +
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
# print(f"Running from {repo_dir}")

from tree_classifications.merge_inputs_outputs import aggregated_metrics
from tree_classifications.sentinel_parallel import sentinel_download

# -

# Allow this file to be run with arguments from the command line
import argparse
import logging
logging.basicConfig(level=logging.INFO)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Run the model predictions in parallel
        
Example usage locally:
python3 predictions_batch.py --csv batch.csv""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--csv", type=str, required=True, help="csv filename containing a list of tiles for this subprocess")
    return parser.parse_args()


# +
# Load the trained model and standard scaler
nearal_network_stub = 'fft_89a_92s_85r_86p'
filename_model = f'/g/data/xe2/cb8590/models/nn_{nearal_network_stub}.keras'
filename_scaler = f'/g/data/xe2/cb8590/models/scaler_{nearal_network_stub}.pkl'

# model = keras.models.load_model(filename_model)
# scaler = joblib.load(filename_scaler)

# Prepare the colour scheme for the tiff files
cmap = {
    0: (240, 240, 240), # Non-trees are white
    1: (0, 100, 0),   # Trees are green
}


# +

def tif_prediction_ds(ds, stub, outdir,  model, scaler, savetif):

    # Calculate vegetation indices
    B8 = ds['nbart_nir_1']
    B4 = ds['nbart_red']
    B3 = ds['nbart_green']
    B2 = ds['nbart_blue']
    ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
    ds['NDVI'] = (B8 - B4) / (B8 + B4)
    ds['GRNDVI'] = (B8 - B3 + B4) / (B8 + B3 + B4)

    # print("Aggregating")
    # Preprocess the temporally and spatially aggregated metrics
    ds_agg = aggregated_metrics(ds)
    ds = ds_agg # I don't think this is necessary since aggregated metrics changes the ds in place
    variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
    ds_selected = ds[variables] 
    ds_stacked = ds_selected.to_array().transpose('variable', 'y', 'x').stack(z=('y', 'x'))

    # print("Normalising")
    # Normalise the inputs using the same standard scaler during training
    X_all = ds_stacked.transpose('z', 'variable').values  # shape: (n_pixels, n_features)
    df_X_all = pd.DataFrame(X_all, columns=ds_selected.data_vars) # Just doing this to silence the warning about not having feature names
    X_all_scaled = scaler.transform(df_X_all)

    # Make predictions and add to the xarray    
    # print("Predicting")
    preds = model.predict(X_all_scaled)
    predicted_class = np.argmax(preds, axis=1)
    pred_map = xr.DataArray(predicted_class.reshape(ds.dims['y'], ds.dims['x']),
                            coords={'y': ds.y, 'x': ds.x},
                            dims=['y', 'x'])
    pred_map.rio.write_crs(ds.rio.crs, inplace=True)

    # print("About to save the tif")
    
    # Save the predictions as a tif file
    da = pred_map.astype('uint8')
    filename = f'{outdir}/{stub}_predicted.tif'
    
    # print("Importing rasterio")
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

    return da

def tif_prediction(tile, outdir='/scratch/xe2/cb8590/Nick_predicted'):
    # Load the sentinel imagery
    with open(tile, 'rb') as file:
        ds = pickle.load(file)
        
    tile_id = "_".join(tile.split('/')[-1].split('_')[:2])
    da = tif_prediction_ds(ds, tile_id, outdir, savetif=True)
    return da

def tif_prediction_bbox(stub, year, outdir, bounds, src_crs,  model, scaler):
    # Run the sentinel download and tree classification for a given location
    ds = sentinel_download(stub, year, outdir, bounds, src_crs)
    da = tif_prediction_ds(ds, stub, outdir, model, scaler, savetif=True)

    # # Trying to avoid memory accumulating with new tiles
    del ds, da
    gc.collect()
    return None

def run_worker(func, rows):
    """Abstracting the for loop & try except for each worker"""
    # Should load this once per worker
    model = keras.models.load_model(filename_model)
    scaler = joblib.load(filename_scaler)

    for row in rows:
        try:
            # mem_before = process.memory_info().rss / 1e9
            func(*row, model, scaler)
            # mem_after = process.memory_info().rss / 1e9
            mem_info = process.memory_full_info()
            print(f"{row[0]}: RSS: {mem_info.rss / 1e9:.2f} GB, VMS: {mem_info.vms / 1e9:.2f} GB, Shared: {mem_info.shared / 1e9:.2f} GB")

            # print(f"{row[0]}: Memory used before {mem_before:.2f} GB, after {mem_after:.2f} GB", flush=True)
        except Exception as e:
            print(f"Error in row {row}:", flush=True)
            traceback.print_exc(file=sys.stdout)

# -

# %%time
if __name__ == '__main__':

    # Load the list of tiles we want to download
    # args = argparse.Namespace(
    #     # csv='/g/data/xe2/cb8590/models/batches/batch_0.csv'
    #     csv='/g/data/xe2/cb8590/models/batches_aus/55HFC.gpkg'
    # )
    args = parse_arguments()
    
    # Download Nick's tiles in serial
    # df = pd.read_csv(args.csv)
    # rows = df['0'].to_list()
    # rows = [[row] for row in rows]
    # run_worker(func, rows)

    # Download the aus ka08 subtiles in serial
    gdf = gpd.read_file(args.csv)
    rows = []
    outdir = '/scratch/xe2/cb8590/ka08_trees'
    for i, tile in gdf.iterrows():
        stub = tile['stub']
        year = "2020"
        bounds = tile['geometry'].bounds
        crs = gdf.crs
        rows.append([stub, year, outdir, bounds, crs])
    func = tif_prediction_bbox

    # rows = rows[:10]

    run_worker(func, rows)