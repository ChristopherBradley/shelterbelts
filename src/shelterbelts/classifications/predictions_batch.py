# +
import os, sys
import argparse
import logging
logging.basicConfig(level=logging.INFO)

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
# Change directory to this repo. Need to do this when using the DEA environment since I can't just pip install -e .
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}/src")
elif os.path.basename(os.getcwd()) != repo_name:
    repo_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from repo root
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)
# print(f"Running from {repo_dir}")

from shelterbelts.classifications.merge_inputs_outputs import aggregated_metrics
from shelterbelts.classifications.sentinel_parallel import sentinel_download

# -

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
    print(f"Saved: {filename}", flush=True)

    return da

def tif_prediction(tile, outdir='/scratch/xe2/cb8590/Nick_predicted'):
    # Load the sentinel imagery
    with open(tile, 'rb') as file:
        ds = pickle.load(file)
        
    tile_id = "_".join(tile.split('/')[-1].split('_')[:2])
    da = tif_prediction_ds(ds, tile_id, outdir, savetif=True)
    return da

def tif_prediction_bbox(stub, year, outdir, bounds, src_crs, model, scaler):
    # Run the sentinel download and tree classification for a given location
    ds = sentinel_download(stub, year, outdir, bounds, src_crs)
    da = tif_prediction_ds(ds, stub, outdir, model, scaler, savetif=True)

    # # Trying to avoid memory accumulating with new tiles
    del ds, da
    gc.collect()
    return None

def run_worker(func, rows, nn_dir='/g/data/xe2/cb8590/models', nn_stub='fft_89a_92s_85r_86p'):
    """Abstracting the for loop & try except for each worker"""
    
    # Would be nice to not hardcode this so other people can use their own models
    filename_model = os.path.join(nn_dir, f'nn_{nn_stub}.keras')
    filename_scaler = os.path.join(nn_dir, f'scaler_{nn_stub}.pkl')

    # Should load this once per worker, so they aren't sharing the same model
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

def predictions_batch(gpkg, outdir, year=2020, nn_dir='/g/data/xe2/cb8590/models', nn_stub='fft_89a_92s_85r_86p', limit=None):
    """Use the model to make tree classifications based on sentinel imagery for that year
    
    Parameters
    ----------
        gpkg: Geopackage with the bounding box for each tile to download. A stub gets automatically assigned based on the center of the bbox.
        outdir: Folder to save the output tifs.
        year: The year of sentinel imagery to use as input for the tree predictions.
        nn_dir: The directory containing the neural network model and scaler.
        nn_stub: The stub of the neural network and preprocessing scaler model to make the predictions.
        limit: The number of rows in the gpkg to read. 'None' means use all the rows.
    
    Downloads
    ---------
        A tif with tree classifications for each bbox in the gpkg
    
    
    """
    
    gdf = gpd.read_file(gpkg)
    crs = gdf.crs
    rows = []
    for i, row in gdf.iterrows():
        bbox = row['geometry'].bounds
        centroid = row['geometry'].centroid
        
        # Maybe I should make it so that if there is a 'stub' column in the gdf then use that, otherwise create a stub automatically like this
        stub = f"{centroid.y:.2f}-{centroid.x:.2f}".replace(".", "_")[1:]
        rows.append([stub, year, outdir, bbox, crs])

    if limit:
        rows = rows[:int(limit)]

    # Legacy argument, not sure when we'd want to use run_worker with another function anymore.
    func = tif_prediction_bbox

    run_worker(func, rows, nn_dir, nn_stub)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gpkg", type=str, required=True, help="filename containing the tiles to use for bounding boxes. Just uses the geometry, and assigns a stub based on the central point")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for the final classified tifs")
    parser.add_argument("--year", type=int, default=2020, help="Year of satellite imagery to download for doing the classification")
    parser.add_argument("--nn_dir", type=str, default='/g/data/xe2/cb8590/models', help="The stub of the neural network model and preprocessing scaler")
    parser.add_argument("--nn_stub", type=str, default='fft_89a_92s_85r_86p', help="The stub of the neural network model and preprocessing scaler")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to process")

    return parser.parse_args()


# +
# %%time
if __name__ == '__main__':

    args = parse_arguments()
    
    gpkg = args.gpkg
    outdir = args.outdir
    year = int(args.year)
    nn_dir = args.nn_dir
    nn_stub = args.nn_stub
    limit = args.limit
    
    predictions_batch(gpkg, outdir, year, nn_dir, nn_stub, limit)



# +
# # %%time
# filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_10.gpkg'
# outdir = '/scratch/xe2/cb8590/tmp'
# predictions_batch(filename, outdir, limit=10)

# # 40 secs for 1 file
# # 6 mins for 10 files
