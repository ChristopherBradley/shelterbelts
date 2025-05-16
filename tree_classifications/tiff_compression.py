# +
# Merge Nick's tiffs into larger files for easier viewing in QGIS

# +
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import ColorInterp
from tqdm import tqdm
import dask

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# -

import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# %%time
# Getting the compression details of tas vegetation
filename_tas_input = "../data/Saving_Tiffs/Tas_WoodyVeg_202403_v2.2.tif"
with rasterio.open(filename_tas_input) as src:
    print("Driver:", src.driver)
    print("Dtype:", src.dtypes[0])
    print("Compression:", src.profile.get('compress'))
    print("Tiled:", src.profile.get('tiled'))
    print("Block size:", (src.profile.get('blockxsize'), src.profile.get('blockysize')))
    print("Width x Height:", src.width, "x", src.height)
    print("Count (bands):", src.count)
    print("CRS:", src.crs)
    overviews = src.overviews(1)  # Check for band 1
    print("Overviews:", overviews)

# %%time
# Load the raster with rioxarray
da = rxr.open_rasterio(filename_tas_input).isel(band=0).drop_vars('band')

# +
# %%time
# Write the raster with rioxarray

# Be careful with rio.to_raster because the default of no compression multiplies the filesize x100
# da.rio.to_raster('../data/Saving_Tiffs/woodyveg_toraster.tif')

# Deflate seems best for untiled
# da.rio.to_raster('../data/Saving_Tiffs/woodyveg_toraster_deflate_tiled.tif', compress='deflate', tiled=True)

# lzw with a large blocksize seems best for tiled
blocksize = 2**10
filename_tas_output = f'../data/Saving_Tiffs/woodyveg_toraster_lzw_{blocksize}.tif'
da.rio.to_raster(filename_tas_output, compress='lzw', tiled=True, 
                 blockxsize=blocksize, blockysize=blocksize)
# !gdaladdo {filename_tas_output} 2 4 8 16 32 64 128
# +
# %%time
# Save the woodyveg with colours
filename = f'../data/Saving_Tiffs/woodyveg_toraster_coloured.tif'
cmap = {
    1: (240, 240, 240), # Non-trees are white
    2: (0, 100, 0),   # Trees are green
    255: (0, 100, 200)  # Nodata is blue
}
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
    tiled=True,
    compress="LZW",
    blockxsize=2**10,
    blockysize=2**10,
    photometric="palette",
) as dst:
    dst.write(da.values, 1)
    dst.write_colormap(1, cmap)
    
# !gdaladdo {filename} 2 4 8 16 32 64

# 25 secs
# -

# %%time
# The simple way without attaching colours
# da_worldcover.rio.to_raster(filename_worldcover_output, compress=compress, tiled=True, 
#                  blockxsize=blocksize, blockysize=blocksize)
# # !gdaladdo {filename_worldcover_output} 2 4 8 16 32 64

# +
filename_worldcover_input = f'../data/Saving_Tiffs/ESA_WorldCover_10m_2021_v200_S33E147_Map.tif'
da_worldcover = rxr.open_rasterio(filename_worldcover_input).isel(band=0).drop_vars('band')

blocksize = 2**11
compress = 'lzw'
filename_worldcover_output = f'../data/Saving_Tiffs/worldcover_toraster_coloured.tif'
filename_worldcover_output

# +
# %%time
# The better way with colours
worldcover_cmap = {
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (250, 0, 0),
    60: (180, 180, 180),
    70: (240, 240, 240),
    80: (0, 100, 200),
    90: (0, 150, 160),
    100: (250, 230, 160)
}
with rasterio.open(
    filename_worldcover_output,
    "w",
    driver="GTiff",
    height=da_worldcover.shape[0],
    width=da_worldcover.shape[1],
    count=1,
    dtype="uint8",
    crs=da_worldcover.rio.crs,
    transform=da_worldcover.rio.transform(),
    tiled=True,
    compress="LZW",
    blockxsize=blocksize,
    blockysize=blocksize,
    photometric="palette",
) as dst:
    dst.write(da_worldcover.values, 1)
    dst.write_colormap(1, worldcover_cmap)
    
# !gdaladdo {filename_worldcover_output} 2 4 8 16 32 64
# -
da_worldcover_subset = da_worldcover.isel(y=slice(20000, 21000), x=slice(20000, 21000))

# Testing compression for a tiff with floating points
filename_tas_input = "../data/Saving_Tiffs/Tas_CanopyCover_202403_v2.2.tif"
da_original = rxr.open_rasterio(filename_tas_input).isel(band=0).drop_vars('band')

# %%time
compress = "LZW"
blocksize = 2**11
filename = f'../data/Saving_Tiffs/canopy_cover_{da_original.dtype}_{compress}_{blocksize}.tif'
da_original.rio.to_raster(filename, compress=compress, tiled=True, 
                  blockxsize=blocksize, blockysize=blocksize, nodata=255)
filename

# +
# %%time
# Experimenting with a smaller area (just 100kmx100km around flinders island)
da_sliced = da_original.isel(y=slice(10000,20000), x=slice(-10001,-1))

da = da_sliced.where(da_sliced <= 1000, np.nan)
da = da/10
da = da.where(~np.isnan(da), 255)
da = da.astype('uint8')
da = da.rio.set_nodata(255)

compress = "LZW"
blocksize = 2**11
filename = f'../data/Saving_Tiffs/canopy_cover_{da_sliced.shape[0]}_{da.dtype}_{compress}_{blocksize}.tif'
da.rio.to_raster(filename, compress=compress, tiled=True, 
                  blockxsize=blocksize, blockysize=blocksize, nodata=255)
filename

# 32MB for uint16
# 43MB for float32 (LZW fast) or 32MB (deflate slow)
# 13MB for uint8

# +
# %%time
# Convert to uint8
da = da_original.where(da_original != 65535, np.nan)
da = da/10
da = da.where(~np.isnan(da), 255)
da = da.astype('uint8')
da = da.rio.set_nodata(255)

# 2 mins 23 secs. Takes a long time just to convert the datatype
# -

compress = "LZW"
blocksize = 2**11
filename = f'../data/Saving_Tiffs/canopy_cover_{da.dtype}_{compress}_{blocksize}.tif'
filename

# +
# %%time
da.rio.to_raster(filename, compress=compress, tiled=True, 
                  blockxsize=blocksize, blockysize=blocksize, nodata=255)

# 22 secs
# -

# Neat how you can see the precision just by printing
print(np.float16(0.12345678901234567890123456789))
print(np.float32(0.12345678901234567890123456789))
print(np.float64(0.12345678901234567890123456789))

# %%time
# !gdaladdo {filename} 2 4 8 16 32 64
# 15 secs
