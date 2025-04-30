# +
# Create a training dataset with satellite imagery inputs and tree cover outputs
# -

import os
import glob
import pickle
import numpy as np
import pandas as pd
import random
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(0)
random.seed(0)

def jittered_grid(ds):
    """Create a grid of coordinates spaced 100m apart, with a random 2 pixel jitter"""
    spacing = 10
    jitter_range = np.arange(-2, 3)  # [-2, -1, 0, 1, 2]

    # Create a coordinate grid
    edge_buffer = spacing//2
    y_inds = np.arange(edge_buffer, ds.sizes['y'] - edge_buffer, spacing)
    x_inds = np.arange(edge_buffer, ds.sizes['x'] - edge_buffer, spacing)

    # Apply jitter
    y_jittered_inds = y_inds + np.random.choice(jitter_range, size=len(y_inds))
    x_jittered_inds = x_inds + np.random.choice(jitter_range, size=len(x_inds))

    # Get actual coordinates
    y_jittered_coords = ds['y'].values[y_jittered_inds]
    x_jittered_coords = ds['x'].values[x_jittered_inds]

    # Get non-time-dependent variables
    aggregated_vars = [var for var in ds.data_vars if 'time' not in ds[var].dims]

    data_list = []
    for y_coord in y_jittered_coords:
        for x_coord in x_jittered_coords:
            values = {var: ds[var].sel(y=y_coord, x=x_coord, method='nearest').item() for var in aggregated_vars}
            values.update({'y': y_coord, 'x': x_coord})
            data_list.append(values)

        
    # Create DataFrame
    df = pd.DataFrame(data_list)
    return df


def tile_csv(tile):
    """Merge the trees from Nick's files and worldcover reprojected"""
    tile_id = "_".join(tile.split('/')[-1].split('_')[:2])

    tree_cover_filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{tile_id}_binary_tree_cover_10m.tiff'
    ds_nick_trees = rxr.open_rasterio(tree_cover_filename).isel(band=0).drop_vars('band').astype(int)

    worldcover_filename = f'/scratch/xe2/cb8590/Nick_worldcover_reprojected/{tile_id}_worldcover.tif'
    ds_worldcover = rxr.open_rasterio(worldcover_filename).isel(band=0).drop_vars('band')
    ds_worldcover_trees = ((ds_worldcover == 10) | (ds_worldcover == 20)).astype(int)  # 10 is trees, and 20 is shrubs
    
    gch_filename = f'/scratch/xe2/cb8590/Nick_GCH/{tile_id}_10m.tif'
    ds_gch = rxr.open_rasterio(gch_filename).isel(band=0).drop_vars('band')
    ds_gch_trees = (ds_gch >= 1).astype(int)

    ds = xr.Dataset({
        "nick_trees": ds_nick_trees,
        "worldcover_trees": ds_worldcover_trees,
        "global_canopy_height_trees": ds_gch_trees
    })

    df = jittered_grid(ds)
    df["tile_id"] = tile_id
    
    return df


# +
# Create a dataframe of imagery and woody veg or canopy cover classifications for each tile
tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
world_cover_dir = "/scratch/xe2/cb8590/Nick_worldcover_reprojected"
global_canopy_height_dir = "/scratch/xe2/cb8590/Nick_GCH"

# outdir = "/scratch/xe2/cb8590/Nick_csv"

# Using the sentinel tiles since these were used to train my model, so I can compare worldcover on the same tiles
sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
sentinel_tiles = glob.glob(f'{sentinel_dir}/*')
print("Number of sentinel tiles:", len(sentinel_tiles))

# Using the GCH tiles because I haven't finished downloading them, so this just uses the ones already downloaded
gch_tiles = glob.glob(f'{global_canopy_height_dir}/*10m.tif')
print("Number of global canopy height tiles:", len(gch_tiles))

# Using the GCH tiles because I haven't finished downloading them, so this just uses the ones already downloaded
worldcover_tiles = glob.glob(f'{world_cover_dir}/*')
print("Number of worldcover tiles:", len(worldcover_tiles))
# -

# Random sample of 1000 tiles so it doesn't take so long to read them all in
sampled_tiles = random.sample(gch_tiles, 1000)

# +
# %%time
# Find the tree status from worldcover, global canopy height and Nick's tiff files for lots of jittered gridded points across lots of files
dfs = []
for tile in sampled_tiles:
    df = tile_csv(tile)
    dfs.append(df)
    
df_all = pd.concat(dfs)

# Took 30 secs for 100, 3 mins for 700, 6 mins for 1000
# -

# Save these tree comparisons
filename = "/g/data/xe2/cb8590/Nick_outlines/nick_vs_gch.csv"
df_all.to_csv(filename, index="False")

df_all['global_canopy_height_trees'].value_counts()

# +
print(classification_report(df_all['nick_trees'], df_all['worldcover_trees']))

# Worldcover has an accuracy of 82% but recall for 1's of only 55% without shrubs or 62% with shrubs. So it misses a lot of trees, as I've noticed qualitatively before.

# +
print(classification_report(df_all['nick_trees'], df_all['global_canopy_height_trees']))

# Global canopy height has an accuracy of 83% with a recall of 65% for 1's. So a little better than worldcover
# -

# Generated in megaregions.py
filename_bioregions = "/g/data/xe2/cb8590/Nick_outlines/centroids_named.gpkg"
