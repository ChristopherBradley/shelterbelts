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
# -

# Write the raster with colours predefined


# Getting the compression details of worldcover
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

# +
# # %%time
# This code makes a file that's 1MB but no colour
# filename_merged = '/scratch/xe2/cb8590/tmp/merged_tree_cover.tif'
# subset.rio.to_raster(
#     filename_merged,
#     tiled=True,
#     blockxsize=256,
#     blockysize=256,
#     compress='deflate'
# )
# print("Saved filename_merged")
# # 300km: 7 mins with dtype float, but only 1 min with dtype uint8 
# # 500km: 5 mins

# +
# %%time
# Save the raster with colour encodings embedded
# This code makes a file that's 1GB but with colour
filename_rasterio = '/scratch/xe2/cb8590/tmp/merged_tree_cover_rasterio.tif'
with rasterio.open(filename_rasterio, "w",
    driver="GTiff",
    height=subset.shape[0],
    width=subset.shape[1],
    count=1,
    dtype="uint8",
    crs=subset.rio.crs,
    transform=subset.rio.transform(),
    photometric='palette') as dst:

    dst.write(subset.values, 1)

    # Define a simple colormap (value: (R, G, B))
    colormap = {
        0: (255, 255, 255),  # white for background
        1: (0, 128, 0),      # green for trees
    }
    dst.write_colormap(1, colormap)
print(filename_rasterio)

# 20 secs to use rasterio when it only took 1 second with 
# -


