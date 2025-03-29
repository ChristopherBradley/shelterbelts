# +
# Super simple methodology to get a shelter score based on distance from shelterbelt and dominant wind direction 
# -

import os
import numpy as np
import rioxarray as rxr
from rasterio.enums import Resampling
from scipy.ndimage import label
import matplotlib.pyplot as plt



outdir = "../data/"
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


# +
# Load wind data


# +
# Decide on dominant wind direction


# +
# Assign each pasture pixel a distance from shelterbelt in the prevailing wind direction


# +
# Save these distances as a tif file

