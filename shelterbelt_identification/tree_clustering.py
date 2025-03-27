# +
# Group trees together and calculate statistics including: 
# Number of shelterbelts w/ length, width, area, perimeter, height (min, mean, max) for each (and then mean and sd)
# Area of sheltered and unsheltered crop & pasture by region and different thresholds
# -
import os
import glob
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import pyproj
from scipy.ndimage import label
import matplotlib.pyplot as plt


# Make the panda displays more informative
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Load the woody veg or canopy cover tiff file
outdir = "../data/"
filename = os.path.join(outdir, "Tas_WoodyVeg_201903_v2.2.tif")  # Binary classifications
sub_stub = "woodyveg"
ds_original = rxr.open_rasterio(filename)

# Create a 5km x 5km bounding box
lon, lat = 147.4793, -42.3906
buffer_m = 2500
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True).transform
x, y = project(lon, lat)
bbox = x - buffer_m, y + buffer_m, x + buffer_m, y - buffer_m  # Not sure why the miny requires addition instead of subtraction, but this gives the 5km x 5km region
minx, miny, maxx, maxy = bbox
bbox

# Select the 5km x 5km region
da = ds_original.sel(band=1, x=slice(minx, maxx), y=slice(miny, maxy))
bool_array = np.array(da.values - 1, dtype = bool)

# Assign a label to each group of pixels
structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) 
shelterbelts, num_features = label(woody_veg, structure)

# List the coords in each group
coord_lists = {i: list(zip(*np.where(shelterbelts == i))) for i in range(1, num_features)}

# +
# Calculate area, width, length for each shelterbelt
group_stats = []
for i, coords in coord_lists.items():
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    
    area = len(coords)
    x_length = max(xs) - min(xs)
    y_length = max(ys) - min(ys)
    stats = {
        'area':area,
        'x_length':x_length,
        'y_length':y_length,
        'max_length':max(x_length, y_length)
        # 'coords':coords
    }
    group_stats.append(stats)

df_shelterbelts = pd.DataFrame(group_stats, index=coord_lists.keys())
# -


# Filter out any shelterbelts less than a certain length, e.g. 100m
length_threshold = 10   # 100m 
df_large_shelterbelts = df_shelterbelts[df_shelterbelts['max_length'] > length_threshold]
large_shelterbelts = shelterbelts.copy()
mask = ~np.isin(shelterbelts, df_large_shelterbelts.index)
large_shelterbelts[mask] = 0

# Add these groups to the original xarray
da_reset = da.reset_coords("band", drop=True)
ds = da_reset.to_dataset(name="woody_veg")
da_shelterbelts = xr.DataArray(
    shelterbelts,
    dims=["y", "x"],
    coords={"x": ds.x, "y": ds.y},
    name="shelterbelts"
)
da_large_shelterbelts = xr.DataArray(
    large_shelterbelts,
    dims=["y", "x"],
    coords={"x": ds.x, "y": ds.y},
    name="large_shelterbelts"
)
ds["shelterbelts"] = da_shelterbelts
ds["large_shelterbelts"] = da_large_shelterbelts

ds['large_shelterbelts'].rio.to_raster("../data/test_large_shelterbelts.tif")

ds['large_shelterbelts'].plot()
