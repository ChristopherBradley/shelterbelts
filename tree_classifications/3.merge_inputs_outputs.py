# +
# Create a training dataset with satellite imagery inputs and tree cover outputs
# -

import os
import glob
import pickle
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage


def aggregated_metrics(ds):
    """Add a temporal median, temporal std, focal mean, and focal std for each temporal band"""
    # Make a list of the variables with a time dimension
    time_vars = [var for var in ds.data_vars if 'time' in ds[var].dims]

    # Calculate aggregated metrics per pixel
    for variable in time_vars:

        # Temporal metrics
        ds_median_temporal = ds[variable].median(dim="time", skipna=True)  # Not sure if I should be doing some kind of outlier removal before this
        ds_std_temporal = ds[variable].std(dim="time", skipna=True)

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

    return ds

def jittered_grid(ds):
    """Create a grid of coordinates spaced 100m apart, with a random 2 pixel jitter"""
    spacing = 10
    jitter_range = np.arange(-2, 3)  # [-2, -1, 0, 1, 2]

    # Create a coordinate grid (starting 5 pixels from the edge)
    y_coords = np.arange(5, ds.sizes['y'] - 5, spacing)
    x_coords = np.arange(5, ds.sizes['x'] - 5, spacing)

    # Apply jitter to each coordinate
    y_jittered_inds = y_coords + np.random.choice(jitter_range, size=len(y_coords))
    x_jittered_inds = x_coords + np.random.choice(jitter_range, size=len(x_coords))

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


# +
# Create a dataframe of imagery and woody veg or canopy cover classifications for each tile
tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
outdir = "/scratch/xe2/cb8590/Nick_csv"

sentinel_tiles = glob.glob(f'{sentinel_dir}/*')

# +
# %%time

dfs = []
for sentinel_tile in sentinel_tiles[:2]:

    tile_id = "_".join(sentinel_tile.split('/')[-1].split('_')[:2])
    tree_cover_filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{tile_id}_binary_tree_cover_10m.tiff'

    # Load the sentinel imagery and tree cover into an xarray
    with open(sentinel_tile, 'rb') as file:
        ds = pickle.load(file)

    # Load the woody veg and add to the main xarray
    ds1 = rxr.open_rasterio(tree_cover_filename)
    ds2 = ds1.isel(band=0).drop_vars('band')
    ds = ds.rio.reproject_match(ds2)
    ds['tree_cover'] = ds2.astype(float)

    # Calculate vegetation indices
    B8 = ds['nbart_nir_1']
    B4 = ds['nbart_red']
    B3 = ds['nbart_green']
    B2 = ds['nbart_blue']
    ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
    ds['NDVI'] = (B8 - B4) / (B8 + B4)
    ds['GRNDVI'] = (B8 - B3 + B4) / (B8 + B3 + B4)

    # Calculate the aggregated metrics
    ds = aggregated_metrics(ds)

    # Remove the temporal bands
    variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
    ds_selected = ds[variables] 

    # Select pixels to use for training/testing
    df = jittered_grid(ds)

    # Change outputs to 0 and 1, instead of 1 and 2
    df['tree_cover'] = df['tree_cover'] - 1

    # Normalize all columns in the DataFrame to be between 0 and 1
    df_normalized = (df - df.min()) / (df.max() - df.min())

    # Save a copy of this dataframe just in case something messes up later (since this is going to take 2 to 4 hours)
    filename = os.path.join(outdir, f"{tile_id}_df_tree_cover.csv")
    df_normalized.to_csv(filename)
    print("Saved", filename)

    dfs.append(df_normalized)

# 30 secs each 
# -

# Combine all the dataframes
df_all = pd.concat(dfs)

# Feather file is more efficient, but csv is more readable
filename = os.path.join(outdir, f"tree_cover_preprocessed.csv")
df_all.to_csv(filename)
print("Saved", filename)
