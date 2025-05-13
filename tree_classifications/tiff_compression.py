# +
# Merge Nick's tiffs into larger files for easier viewing in QGIS
# -

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





