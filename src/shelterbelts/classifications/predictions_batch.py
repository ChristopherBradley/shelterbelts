# +
import os, sys
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("rasterio").setLevel(logging.ERROR)

import pickle
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import nearest_points

import xarray as xr
import rioxarray as rxr
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
import joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Remove tensorflow logging info


import gc
import psutil
process = psutil.Process(os.getpid())


# +
# Change directory to this repo. Need to do this when using the DEA environment since I can't just pip install -e .
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
src_dir = os.path.join(repo_dir, 'src')
os.chdir(src_dir)
sys.path.append(src_dir)
# print(src_dir)

from shelterbelts.classifications.merge_inputs_outputs import aggregated_metrics
from shelterbelts.classifications.sentinel_nci import download_ds2_bbox  # Will probably have to create a predictions_nci, and predictions_dea to avoid datacube import issues

# -

# Prepare the colour scheme for the tiff files
cmap_binary = {
    0: (240, 240, 240), # Non-trees are white
    1: (0, 100, 0),   # Trees are green
}

# Loading this once instead of every time in tif_prediction_ds. Could instead load once in predictions_batch and pass as an argument
gdf_koppen = gpd.read_file('/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned2.gpkg')


# +
def tif_prediction_ds(ds, outdir, stub, model, scaler, savetif, add_xy=True, confidence=False, model_weightings=None):

    # Calculate vegetation indices
    B8 = ds['nbart_nir_1']
    B4 = ds['nbart_red']
    B3 = ds['nbart_green']
    B2 = ds['nbart_blue']
    ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))  # Should look into why this is giving NaN subtraction warnings
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
    
    # The old models didn't include the xy coords, but the new ones do because it improved accuracy by around 0.5%. 
    # Similar improvement in accuracy by adding a 1-hot encoded koppen class, but no benefit to having both xy and koppen.
    if add_xy:
        # Add x, y coordinates
        y, x = ds_stacked['z'].to_index().levels  
        coords = pd.DataFrame(ds_stacked['z'].to_index().tolist(), columns=['y', 'x'])
        df_X_all = pd.concat([df_X_all, coords], axis=1)

        # Reproject to epsg:4326. Might have been better to have trained the model on EPSG:3857 to avoid this
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        df_X_all['x'], df_X_all['y'] = transformer.transform(df_X_all['x'].values, df_X_all['y'].values)

    # Attach koppen categories
    # I should be able to figure out which model to use based on a single central point before we call this function
    # gdf_points = gpd.GeoDataFrame(
    #     df_X_all,
    #     geometry=gpd.points_from_xy(df_X_all['x'], df_X_all['y']),
    #     crs="EPSG:4326"
    # )
    # gdf_joined = gpd.sjoin(gdf_points, gdf_koppen, how='left', predicate='within')
    # df_X_all['Koppen'] = gdf_joined['Name'].values
    
    # Convert to float32 to match datatypes used when training
    df = df_X_all
    for col in df.select_dtypes(include=['float64']).columns:
        # df[col] = df[col].astype(np.float16)
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int16)
    
    # import pdb; pdb.set_trace()
    
    # This happens when there's no good data for that location (e.g. always clouds, or always in a dark shadow from a north-south ridgeline)
    # bad = ~np.isfinite(df.to_numpy())
    df = df.replace([np.inf, -np.inf], np.nan)  
    df = df.fillna(df.median())  # Makes the blue mountains bug not quite as bad, but still bad.
    df = df.fillna(0)  # Sometimes the focal_std is all NaN. If we fill with 1 or 100 then all the predictions become 0. Filling with 0 seems to work nicely.

    if model_weightings:
        # Make a bunch of predictions from the ensemble of models and take a weighted average.
        all_preds = []
        weights = []
        for key in model_weightings:
            model, scaler, weighting = model_weightings[key]
            X_all_scaled = scaler.transform(df)
            model_preds = model.predict(X_all_scaled)
            all_preds.append(model_preds * weighting)
            weights.append(weighting)
        preds = np.sum(all_preds, axis=0) / np.sum(weights)    
        
    else:
        # Make predictions and add to the xarray    
        X_all_scaled = scaler.transform(df)
        preds = model.predict(X_all_scaled)
    
    if confidence:
        cmap_BrBG = plt.cm.BrBG
        norm = mcolors.Normalize(vmin=0, vmax=100)
        cmap = {i: tuple(int(c*255) for c in cmap_BrBG(norm(i))[:3]) for i in range(101)}
        # predicted_class = (preds[:,1] * 100).astype('uint8')  # This floors the results, whereas we want to round them
        predicted_class = np.rint(preds[:, 1] * 100).astype('uint8')
    else:
        cmap = cmap_binary
        predicted_class = np.argmax(preds, axis=1)  
    
    pred_map = xr.DataArray(predicted_class.reshape(ds.sizes['y'], ds.sizes['x']),
                            coords={'y': ds.y, 'x': ds.x},
                            dims=['y', 'x'])
    pred_map.rio.write_crs(ds.rio.crs, inplace=True)

    # print("About to save the tif")
    
    # Save the predictions as a tif file
    da = pred_map.astype('uint8')
    filename = f'{outdir}/{stub}_predicted.tif'
    os.makedirs(outdir, exist_ok=True)
    
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
        # photometric="palette",
    ) as dst:
        dst.write(da.values, 1)
        dst.write_colormap(1, cmap)
    print(f"Saved: {filename}", flush=True)

    return da

def tif_prediction(sentinel_filename, outdir, model_filename, scaler_filename, savetif=True):
    """Predict unknown data"""
    with open(sentinel_filename, 'rb') as file:
        ds = pickle.load(file)
        
    model = keras.models.load_model(model_filename)
    scaler = joblib.load(scaler_filename)
        
    # tile_id = "_".join(tile.split('/')[-1].split('_')[:2])
    tile_id = sentinel_filename.split('/')[-1].split('.')[0]
    da = tif_prediction_ds(ds, outdir, tile_id, model, scaler, savetif)
    return da

def tif_prediction_bbox(stub, year, outdir, bounds, src_crs, model, scaler, confidence=False, model_weightings=None):
    """Run the sentinel download and tree classification for a given location"""
    # from shelterbelts.classifications.sentinel_nci import download_ds2_bbox  # Will probably have to create a predictions_nci, and predictions_dea to avoid datacube import issues

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    ds = download_ds2_bbox(bounds, start_date, end_date, outdir, stub, save=False, input_crs=src_crs) # If we do save all the sentinel pickle files, it took about 20TB for all of NSW.

    da = tif_prediction_ds(ds, outdir, stub, model, scaler, savetif=True, confidence=confidence, model_weightings=None)

    # # Trying to avoid memory accumulating with new tiles
    del ds 
    del da
    gc.collect()
    return None

def run_worker(rows, nn_dir='/g/data/xe2/cb8590/models', nn_stub='fft_89a_92s_85r_86p', multi_model=False, confidence=False):
    """Abstracting the for loop & try except for each worker"""
    model, scaler, model_weightings = None, None, None

    # Loading this once per worker, so they aren't sharing the same model
    if not multi_model:
        filename_model = os.path.join(nn_dir, f'nn_{nn_stub}.keras')
        filename_scaler = os.path.join(nn_dir, f'scaler_{nn_stub}.pkl')
        model = keras.models.load_model(filename_model)
        scaler = joblib.load(filename_scaler)
    else:
        koppen_classes = gdf_koppen['Name'].unique()
        koppen_classes = list(koppen_classes) + ['all'] # Add the overall model too, in case some regions don't land in a specific class
        model_dict = dict()
        for koppen_class in koppen_classes:
            filename_model = os.path.join(nn_dir, f'nn_{nn_stub}_{koppen_class}.keras')
            filename_scaler = os.path.join(nn_dir, f'scaler_{nn_stub}_{koppen_class}.pkl')
            model = keras.models.load_model(filename_model)
            scaler = joblib.load(filename_scaler)
            model_dict[koppen_class] = (model, scaler)

    for row in rows:
        try:
            if multi_model:
                
                # Choose which of the 6 models to use based on the center coordinate
                bbox = row[3]
                center = (bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2
                point = Point(center)

#                 # Using the nearest polygon instead of contains because the koppen boundaries miss quite a bit of landmass near the coastlines
#                 nearest_geom = min(gdf_koppen.geometry, key=lambda g: point.distance(g))
#                 match = gdf_koppen[gdf_koppen.geometry == nearest_geom]

#                 if len(match) > 0:
#                     koppen_class = match.iloc[0]['Name']
#                 else:
#                     koppen_class = 'all'  # There should always be a match now that I'm using the nearest polygon
#                 model = model_dict[koppen_class][0]
#                 scaler = model_dict[koppen_class][1]
#                 print(f"predicting with model: {koppen_class}")

                # Find which models are within a given distance of this pixel
                gdf_koppen["distance"] = gdf_koppen.geometry.distance(point)
                distance_degree_threshold = 1
                distance_km = distance_degree_threshold * 100   # Rough conversion of 1 degree = 100km
                chosen_models = gdf_koppen[gdf_koppen['distance'] < distance_degree_threshold]  # Only using models if the point is within 1 degree of that polygon
                chosen_models = chosen_models[['Name', 'distance']].sort_values("distance", ascending=False).reset_index()
                n_classes = len(chosen_models)

                # Assign weightings based on how close to the polygon this pixel is
                remaining_percentage = 100
                model_weightings = dict()  # {stub: (model, scaler, weighting)}
                for i, chosen_row in chosen_models.iterrows():
                    koppen_class = chosen_row['Name']
                    if i == n_classes - 1:
                        model_weightings[koppen_class] = (model_dict[koppen_class][0], model_dict[koppen_class][1], remaining_percentage)
                    else:
                        weighting = (100/n_classes) - (chosen_row['distance'] * distance_km)/n_classes  
                        model_weightings[koppen_class] = (model_dict[koppen_class][0], model_dict[koppen_class][1], weighting)
                        remaining_percentage = remaining_percentage - weighting
                print(f"Predicting {row[0]} with these model weights: {[(k, model_weightings[k][2]) for k in model_weightings]}")
                    
            # mem_before = process.memory_info().rss / 1e9
            tif_prediction_bbox(*row, model, scaler, confidence=confidence, model_weightings=model_weightings)
            # mem_after = process.memory_info().rss / 1e9
            mem_info = process.memory_full_info()
            print(f"{row[0]}: RSS: {mem_info.rss / 1e9:.2f} GB, VMS: {mem_info.vms / 1e9:.2f} GB, Shared: {mem_info.shared / 1e9:.2f} GB")

            # print(f"{row[0]}: Memory used before {mem_before:.2f} GB, after {mem_after:.2f} GB", flush=True)
        except Exception as e:
            print(f"Error in row {row}:", flush=True)
            traceback.print_exc(file=sys.stdout)

# -

def predictions_batch(gpkg, outdir, year=2020, nn_dir='/g/data/xe2/cb8590/models', nn_stub='fft_89a_92s_85r_86p', limit=None, multi_model=False, confidence=False):
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
        stub = f"{stub}_y{year}"
        rows.append([stub, year, outdir, bbox, crs])

    if limit:
        rows = rows[:int(limit)]
    
    run_worker(rows, nn_dir, nn_stub, multi_model, confidence)


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
    parser.add_argument("--multi_model", action="store_true", help="Use a separate model for each koppen region. Default: False")
    parser.add_argument("--confidence", action="store_true", help="Output a percentage likelihood that it's a tree, instead of a binary label. Default: False")

    return parser.parse_args()


# %%time
if __name__ == '__main__':

    args = parse_arguments()
    
    gpkg = args.gpkg
    outdir = args.outdir
    year = int(args.year)
    nn_dir = args.nn_dir
    nn_stub = args.nn_stub
    limit = args.limit
    multi_model = args.multi_model
    
    predictions_batch(gpkg, outdir, year, nn_dir, nn_stub, limit, multi_model, args.confidence)

# +
# # %%time
# filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_10.gpkg'
# outdir = '/scratch/xe2/cb8590/tmp'
# predictions_batch(filename, outdir, limit=1)

# # # 40 secs for 1 file
# # # 6 mins for 10 files

# +
# # %%time
# filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_10.gpkg'
# outdir = '/scratch/xe2/cb8590/tmp'
# nn_dir = '/g/data/xe2/cb8590/models'
# nn_stub = '4326_float32_s4'
# year = 2020
# limit = 10
# predictions_batch(filename, outdir, year, nn_dir, nn_stub, limit, multi_model=True)

# # # 40 secs for 1 file
# # # 6 mins for 10 files

# +
# # %%time
# filename = '/scratch/xe2/cb8590/tmp/blue_mountains_bad.gpkg'
# outdir = '/scratch/xe2/cb8590/tmp'
# predictions_batch(filename, outdir, nn_stub='4326_float32_s4_all', confidence=True, year=2018, limit=1)

# # # 40 secs for 1 file
# # # 6 mins for 10 files

# +
# # %%time
# filename = '/scratch/xe2/cb8590/tmp/blue_mountains_bad.gpkg'
# outdir = '/scratch/xe2/cb8590/tmp'
# predictions_batch(filename, outdir, nn_stub='4326_float32_s4', confidence=True, year=2018, multi_model=True)

# # # 40 secs for 1 file
# # # 6 mins for 10 files

# +
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 100)
# df = pd.read_csv('/scratch/xe2/cb8590/tmp/blue_mountains_bad.csv')
# df2 = pd.read_csv('/scratch/xe2/cb8590/tmp/df_barra_bboxs_sample.csv')
# df.describe()
# df.isna().sum().to_frame('NaN_count').assign(
#     NaN_percent=lambda x: 100 * x['NaN_count'] / len(df)
# )
