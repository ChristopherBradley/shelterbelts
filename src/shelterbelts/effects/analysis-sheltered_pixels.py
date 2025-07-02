# +
# Super simple methodology to get a shelter score based on distance from shelterbelt and dominant wind direction 
# -

import os
import sys
import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Change directory to the PaddockTS repo
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/shelterbelts")
elif os.path.basename(os.getcwd()) != "shelterbelts":
    paddockTS_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from PaddockTS root
    paddockTS_dir = os.getcwd()
os.chdir(paddockTS_dir)
sys.path.append(paddockTS_dir)
print(paddockTS_dir)

from shelterbelts.apis.barra_daily import wind_dataframe

outdir = "data"
stub = "Fulham"

# Load worldcover
filename = os.path.join(outdir, f"{stub}_worldcover.tif")
da_worldcover = rxr.open_rasterio(filename).squeeze(dim="band", drop=True)
ds = da_worldcover.to_dataset(name="worldcover")
ds['worldcover'].values.shape


# Load the global canopy height
filename = os.path.join(outdir, f"{stub}_canopy_height.tif")
da_canopy_height = rxr.open_rasterio(filename).squeeze(dim="band", drop=True)
da_canopy_height.values.shape


# Align global canopy height to worldcover
ds['canopy_height'] = da_canopy_height.rio.reproject_match(ds, resampling=Resampling.max)


# Cluster shelterbelts
structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) 
shelterbelts, num_features = label(ds['canopy_height'] >= 1, structure)


# +
# Select shelterbelts longer than a given length
coord_lists = {i: list(zip(*np.where(shelterbelts == i))) for i in range(1, num_features)}
x_lengths = {i: max(c[0] for c in coords) - min(c[0] for c in coords) for i, coords in coord_lists.items()}
y_lengths = {i: max(c[1] for c in coords) - min(c[1] for c in coords) for i, coords in coord_lists.items()}
max_lengths = {i: max(x_length, y_length) for (i, x_length), y_length in zip(x_lengths.items(), y_lengths.values())}

threshold = 10 # 100m
large_shelterbelt_indices = [i for i, max_length in max_lengths.items() if max_length > threshold]
mask = ~np.isin(shelterbelts, large_shelterbelt_indices)
large_shelterbelts = shelterbelts.copy()
large_shelterbelts[mask] = 0
# -


# Load wind data
filename = os.path.join(outdir, f"{stub}_barra_daily.nc")
ds_barra = xr.open_dataset(filename)


# Decide on dominant wind direction
df, max_speed, max_direction = wind_dataframe(ds_barra)
df_20km_plus = df.loc['20-30km/hr'] + df.loc['30+ km/hr']
direction_20km_plus = df_20km_plus.index[df_20km_plus.argmax()]
direction_20km_plus


# Assign each pasture pixel a distance from shelterbelt in the prevailing wind direction
plt.imshow(large_shelterbelts)


# Save these distances as a tif file

