# +
# Create a training dataset with satellite imagery inputs and woody veg outputs
# -

import os
import glob
import pickle
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage

outdir = "/g/data/xe2/cb8590/shelterbelts/"

tiles = glob.glob("/g/data/xe2/cb8590/shelterbelts/*_ds2.pkl")

# Testing with a single tile for now. Later put this in a loop for all tiles
tile = tiles[0]
stub = tile.replace(outdir,"").replace("_ds2.pkl","")

# Load the imagery
with open(tile, 'rb') as file:
    ds = pickle.load(file)

# Load the woody veg and add to the main xarray
filename = os.path.join(outdir, f"{stub}_woodyveg_2019.tif")
ds2 = rxr.open_rasterio(filename)
ds['woody_veg'] = ds2.isel(band=0).drop_vars('band')

# +
# Calculate vegetation indices used by Stewart et al. 2025 and Brandt et al. 2023 (NDVI, EVI, GNDVI, BSI, MSAVI)
B8 = ds['nbart_nir_1']
B4 = ds['nbart_red']
B3 = ds['nbart_green']
B2 = ds['nbart_blue']

ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
ds['NDVI'] = (B8 - B4) / (B8 + B4)
ds['GRNDVI'] = (B8 - B3 + B4) / (B8 + B3 + B4)
ds['BSI'] = (B2 + B4 - B3) / (B2 + B4 - B3)
ds['MSAVI2'] = (2 * B8 + 1 - ((2 * B8 + 1) ** 2 - 8 * (B8 - B4) ** 2) ** 0.5) / 2
# -

# Make a list of the variables with a time dimension
time_vars = [var for var in ds.data_vars if 'time' in ds[var].dims]

# Calculate the median and standard deviation for one of these bands (later apply to all)
ds_median = ds['nbart_red'].median(dim="time", skipna=True)
ds_std = ds['nbart_red'].std(dim="time", skipna=True)

# Guessing the focal metrics get applied to the aggregated temporal metrics
radius = 3
kernel_size = 2 * radius + 1 # Guessing the radius doesn't include the center pixel
red_band = ds['nbart_red']
focal_mean = xr.apply_ufunc(
    ndimage.uniform_filter, 
    red_band, 
    kwargs={'size': kernel_size, 'mode': 'nearest'}
)

# Compute focal standard deviation
focal_std = xr.apply_ufunc(
    ndimage.generic_filter, 
    red_band, 
    kwargs={'function': lambda x: x.std(), 'size': kernel_size, 'mode': 'nearest'}
)
focal_std
