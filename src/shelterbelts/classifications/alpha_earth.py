# # Change directory to this repo - this should work on gadi or locally via python or jupyter.
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

from shelterbelts.apis.worldcover import tif_categorical, worldcover_labels

import ee
import rioxarray as rxr
import pandas as pd
import numpy as np
import geemap


ee.Authenticate()
ee.Initialize()

# Load the tree tif
filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/g2_26729_binary_tree_cover_10m.tiff'
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
da_4326 = da.rio.reproject('EPSG:4326')
bbox = da_4326.rio.bounds()

year = 2020

# Prep the embeddings
start_date = f"{year}-01-01"
end_date = f"{year}-12-31"
polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
roi = ee.Geometry.Polygon([polygon_coords])
collection = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterBounds(roi)
    .filterDate(start_date, end_date)
)
image = ee.Image(collection.first())  # Just one image if we just get a single year
# %%time
# Download the embeddings
np_array = geemap.ee_to_numpy(image, region=roi, scale=10)  # Took 20 secs for a 1km x 1km region for 2020

# Create an xarray to align with the tree tif
ny, nx, n_bands = np_array.shape
x_coords = np.linspace(bbox[0], bbox[2], nx)
y_coords = np.linspace(bbox[3], bbox[1], ny)  # top → bottom
ae_da = xr.DataArray(
    np_array, 
    dims=("y", "x", "band"), 
    coords={"x": x_coords, "y": y_coords, "band": np.arange(n_bands)}, 
    name="alpha_earth"
)
ae_da = ae_da.rio.write_crs("EPSG:4326")
ae_da = ae_da.transpose("band", "y", "x")
ae_da_match = ae_da.rio.reproject_match(da)

# Download an example embedding to check it aligns nicely
filename = '/scratch/xe2/cb8590/tmp/ae_da_match2.tif'
ae_da_match.isel(band=0).rio.to_raster(filename)
print(filename)

ae_da_match.shape

da.shape

# Prepare the flattened tree outputs
tree_array = da.values  # shape (y, x), 0/1 labels
lon = da['x'].values
lat = da['y'].values
xx, yy = np.meshgrid(lon, lat)
coords = np.column_stack([xx.ravel(), yy.ravel()])
tree_flat = tree_array.ravel()

# Combine into DataFrame
n_bands, height, width = np_array.shape
inputs_flat = np_array.reshape(-1, n_bands)
tree_flat = da.values.ravel()
columns = [f"emb_{i}" for i in range(n_bands)]
df = pd.DataFrame(inputs_flat, columns=columns)
df['tree'] = tree_flat
