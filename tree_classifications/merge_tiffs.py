# +
# Merge Nick's tiffs into larger files for easier viewing in QGIS
# -

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from tqdm import tqdm
import dask

# Create a dataframe of imagery and woody veg or canopy cover classifications for each tile
tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
filename_bbox = "/g/data/xe2/cb8590/Nick_outlines/nick_bbox_crs.csv"
df_bbox = pd.read_csv(filename_bbox)

df_bbox['crs'].value_counts()

tile_ids = df_bbox[df_bbox['crs'] == 'EPSG:28355']['Tile'].values
tile_ids = tile_ids[:5000]

# %%time
# Load all the tiles
tiles = []
for tile_id in tile_ids:
    tree_cover_filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{tile_id}_binary_tree_cover_10m.tiff'
    tile = rxr.open_rasterio(tree_cover_filename).isel(band=0).drop_vars('band').astype("uint8")
    tiles.append(tile)

# Compute global bounds from all tiles
all_bounds = [tile.rio.bounds() for tile in tiles]
minx = min(b[0] for b in all_bounds)
miny = min(b[1] for b in all_bounds)
maxx = max(b[2] for b in all_bounds)
maxy = max(b[3] for b in all_bounds)

# +
# Assume square pixels, consistent resolution
res_x, res_y = tiles[0].rio.resolution()
res_y = abs(res_y)  # make positive

# Calculate output shape
width = int(np.ceil((maxx - minx) / res_x))
height = int(np.ceil((maxy - miny) / res_y))

# Create output coordinates
x_coords = minx + res_x / 2 + np.arange(width) * res_x
y_coords = maxy - res_y / 2 - np.arange(height) * res_y

# Preallocate output array
merged_data = np.zeros((height, width), dtype=tiles[0].dtype)
# -

# %%time
# Insert each tile
for tile in tiles:
    bx_min, by_min, bx_max, by_max = tile.rio.bounds()

    # Calculate pixel offsets
    x_start = int((bx_min - minx) / res_x)
    y_start = int((maxy - by_max) / res_y)

    # Insert tile data
    data = tile.data
    merged_data[y_start:y_start + data.shape[-2], x_start:x_start + data.shape[-1]] = data

# %%time
# Wrap in xarray.DataArray
merged = xr.DataArray(
    merged_data,
    coords={"y": y_coords, "x": x_coords},
    dims=("y", "x"),
    attrs=tiles[0].attrs
)
merged.rio.write_crs(tiles[0].rio.crs, inplace=True)

diameter = 30000
radius = diameter//2
subset = merged.sel(
    x=slice(merged.x.values[merged.x.size // 2 - radius],
            merged.x.values[merged.x.size // 2 + radius]),
    y=slice(merged.y.values[merged.y.size // 2 - radius],
            merged.y.values[merged.y.size // 2 + radius])
)

# +
# %%time
subset.rio.to_raster(
    filename_merged,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    compress='deflate'
)

# 20km: 400ms 
# 30km: 5 secs
# 100km: 20 secs
# 300km: 7 mins

# +
# %%time
# Turn into Dask array with chunks
merged_dask = merged.chunk({'x': 10, 'y': 10})

merged_dask.rio.to_raster(
    filename_merged,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    compress='deflate'
)
