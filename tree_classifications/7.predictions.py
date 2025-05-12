# Make a prediction of tree cover on each tile using the trained model

# +
import glob
import pickle
import ast
import traceback

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import transform

import pyproj
from pyproj import Transformer

import rasterio
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

# +
# Load the sentinel imagery
with open(tile, 'rb') as file:
    ds = pickle.load(file)
    
# Preprocess the temporally and spatially aggregated metrics
ds_agg = aggregated_metrics(ds)
variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
ds_selected = ds[variables] 
# -

# Normalise the inputs using the same standard scaler during training
# feature_vars = ['var1', 'var2', ..., 'var40'] 
ds_stacked = ds.to_array().transpose('variable', 'y', 'x').stack(z=('y', 'x'))
X_all = ds_stacked.transpose('z', 'variable').values  # shape: (n_pixels, n_features)
X_all_scaled = scaler.transform(X_all)

# Make predictions and add to the xarray
preds = model.predict(X_all_scaled, batch_size=1024)  # shape: (n_pixels, n_classes)
predicted_class = np.argmax(preds, axis=1)
pred_map = xr.DataArray(predicted_class.reshape(ds.dims['y'], ds.dims['x']),
                        coords={'y': ds.y, 'x': ds.x},
                        dims=['y', 'x'])
ds['predictions'] = pred_map



filename = '/g/data/ka08/ga/ga_s2bm_ard_3/51/JYG/2021/06/11/20210611T041036/ga_s2bm_nbart_3-2-1_51JYG_2021-06-11_final_band04.tif'

filename = '/g/data/ka08/catalog/v2/data/part-00000-69927f0b-a852-4def-b0ea-beb5139840f0-c000.snappy.parquet'

df = pd.read_parquet(filename)

df.shape

df['file_uri'].iloc[50000]

df.iloc[50000]['geometry']


# Function to convert GeoJSON dict to a Shapely geometry and reproject to a target CRS
def convert_and_reproject(geom_dict, source_crs, target_crs='EPSG:7844'):
    if not geom_dict:  # handle missing geometries
        return None
    geom = shape(geom_dict)
    if source_crs != target_crs:
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform
        geom = transform(transformer, geom)
    return geom


# %%time
target_crs = 'EPSG:4326'
df['geometry_obj'] = df.apply(
    lambda row: convert_and_reproject(row['geometry'], row['crs'], target_crs=target_crs), axis=1
)
# Took 23 mins, so best not to do that again

# These tiles do indeed match up with the sentinel boundaries. 
unique_geometries = df.drop_duplicates(subset='geometry_obj')['geometry_obj']
gdf = gpd.GeoDataFrame(geometry=unique_geometries)
gdf.set_crs(target_crs, inplace=True)
filename_ka08 = '/g/data/xe2/cb8590/models/ka08_catalog_00000.gpkg'
gdf.to_file(filename_ka08, layer='geometries', driver="GPKG")
