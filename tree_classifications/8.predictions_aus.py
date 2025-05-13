import glob
import math
import pandas as pd
import geopandas as gpd
import subprocess
from shapely.geometry import box
from tensorflow import keras
import joblib
import xarray as xr
import scipy.ndimage as ndimage




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

# Load the sentinel imagery tiles
filename_sentinel_bboxs = "/g/data/xe2/cb8590/Nick_outlines/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
gdf = gpd.read_file(filename_sentinel_bboxs)

test_tile_id = "55HFC"
test_tile = gdf.loc[gdf['Name'] == test_tile_id, 'geometry'].values[0]


polygon = test_tile

# Get centroid of the polygon (in degrees)
centroid = polygon.centroid
lon, lat = centroid.x, centroid.y



if 'ds' in locals():
    del ds

# +
# %%time
# Create a small area in the center for testing
half_deg = 0.02  # 0.01° / 2
bbox = box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)

stub = "test"
outdir = "/scratch/xe2/cb8590/tmp"
year="2020"
bounds = bbox.bounds
src_crs = "EPSG:4326"
ds = sentinel_download(stub, year, outdir, bounds, src_crs)

# Should get a start and finish memory to see how much additional it used
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")  

# +
# sentinel_download with Large compute (7 cores, 32GB)
# 1km x 1km x 1 year = 30, 10, 20, 10, 10, 20s 5GB, 26s 500MB
# 2km x 2km x 1 year = 10, 10, 10, 23s 5GB 
# 4kmx4km = 20s 1.5GB
# 10kmx10km = 34, 25s 5GB, 15s 5GB, 16s 5GB 
# 20km x 20km x 1 year = 69 kernel died, 29s 12GB, 29s 20GB, 30s 20GB, 28s 20GB (but there seems to be 10GB not being used)

# +
# del ds
# gc.collect()
# print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# +
# %%time
# How much time & memory does the sentinel download use compared to the machine learning predictions?
da = tif_prediction_ds(ds, "Test", outdir="/scratch/xe2/cb8590/tmp/", savetif=False)

print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB") 

# +
# Seems like the limiting factor is the sentinel download, not the other preprocessing or machine learning predictions
# 1km x 1km = 3s
# 2km x 2km = 49s, 50s
# 4km x 4km = 60 secs
# 10km x 10km = 
# 20km x 20km = Got stuck > 10 mins
# -

# Seeing which part of the classification is taking so much time
# Load the trained model and standard scaler
filename_model = '/g/data/xe2/cb8590/models/nn_89a_92s_85r_86p.keras'
filename_scaler = '/g/data/xe2/cb8590/models/scaler_89a_92s_85r_86p.pkl'
model = keras.models.load_model(filename_model)
scaler = joblib.load(filename_scaler)

# Calculate vegetation indices
B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B3 = ds['nbart_green']
B2 = ds['nbart_blue']
ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
ds['NDVI'] = (B8 - B4) / (B8 + B4)
ds['GRNDVI'] = (B8 - B3 + B4) / (B8 + B3 + B4)

time_vars = [var for var in ds.data_vars if 'time' in ds[var].dims]


variable = time_vars[0]
variable

time_vars

# Temporal metrics
ds_median_temporal = ds[variable].median(dim="time", skipna=True)  # Not sure if I should be doing some kind of outlier removal before this
ds_std_temporal = ds[variable].std(dim="time", skipna=True)


# %%time
radius = 3
kernel_size = 2 * radius + 1  # 7 pixel diameter because I'm guessing the radius doesn't include the center pixel
ds_mean_focal_7p = xr.apply_ufunc(
    ndimage.uniform_filter, 
    ds_median_temporal, 
    kwargs={'size': kernel_size, 'mode': 'nearest'}
)


import numpy as np
from scipy.signal import fftconvolve


# +
def focal_std_fft(array, kernel):
    kernel = kernel / kernel.sum()  # Normalize to get mean
    mean = fftconvolve(array, kernel, mode='same')
    mean_sq = fftconvolve(array**2, kernel, mode='same')
    std = np.sqrt(mean_sq - mean**2)
    return std

def make_circular_kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(float)



# -

# Using fft convolve greatly improves performance. I just need to double check the outputs look about the same
# %%time
kernel = make_circular_kernel(radius=3)  # Or any other kernel shape
std_array = focal_std_fft(ds_median_temporal.values, kernel)




ds_std_focal_7p = xr.apply_ufunc(
    ndimage.generic_filter, 
    ds_median_temporal, 
    kwargs={'function': lambda x: x.std(), 'size': kernel_size, 'mode': 'nearest'}
)

# +

# Focal metrics
radius = 3
kernel_size = 2 * radius + 1  # 7 pixel diameter because I'm guessing the radius doesn't include the center pixel
ds_mean_focal_7p = xr.apply_ufunc(
    ndimage.uniform_filter, 
    ds_median_temporal, 
    kwargs={'size': kernel_size, 'mode': 'nearest'}
)
ds_std_focal_7p = xr.apply_ufunc(
    ndimage.generic_filter, 
    ds_median_temporal, 
    kwargs={'function': lambda x: x.std(), 'size': kernel_size, 'mode': 'nearest'}
)
ds[f"{variable}_temporal_median"] = ds_median_temporal
ds[f"{variable}_temporal_std"] = ds_std_temporal
ds[f"{variable}_focal_mean"] = ds_mean_focal_7p
ds[f"{variable}_focal_std"] = ds_std_focal_7p
# -

# %%time
ds_agg = aggregated_metrics(ds)


# +



print("Aggregating")
# Preprocess the temporally and spatially aggregated metrics
variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
ds_selected = ds[variables] 
ds_stacked = ds_selected.to_array().transpose('variable', 'y', 'x').stack(z=('y', 'x'))

print("Normalising")
# Normalise the inputs using the same standard scaler during training
X_all = ds_stacked.transpose('z', 'variable').values  # shape: (n_pixels, n_features)
df_X_all = pd.DataFrame(X_all, columns=ds_selected.data_vars) # Just doing this to silence the warning about not having feature names
X_all_scaled = scaler.transform(df_X_all)

# Make predictions and add to the xarray    
print("Predicting")
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

# -




