# +
# Create a training dataset with satellite imagery inputs and woody veg outputs
# -

import os
import glob
import pickle
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage

# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 100)

outdir = "/g/data/xe2/cb8590/shelterbelts/"

tiles = glob.glob("/g/data/xe2/cb8590/shelterbelts/*_ds2.pkl")

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
    y_jittered = y_coords + np.random.choice(jitter_range, size=len(y_coords))
    x_jittered = x_coords + np.random.choice(jitter_range, size=len(x_coords))

    # Extract values at jittered coordinates
    data_list = []
    for y in y_jittered:
        for x in x_jittered:
            values = {var: ds[var].isel(y=y, x=x).item() for var in variables}
            values.update({'y': y, 'x': x})  # Store coordinates
            data_list.append(values)

    # Create DataFrame
    df = pd.DataFrame(data_list)
    return df


# %%time
# Create a dataframe of imagery and woody veg classifications for each tile
dfs = []
for tile in tiles:
    stub = tile.replace(outdir,"").replace("_ds2.pkl","")

    # Load the imagery
    with open(tile, 'rb') as file:
        ds = pickle.load(file)

    # Load the woody veg and add to the main xarray
    filename = os.path.join(outdir, f"{stub}_woodyveg_2019.tif")
    ds1 = rxr.open_rasterio(filename)
    ds['woody_veg'] = ds1.isel(band=0).drop_vars('band')

    # Calculate vegetation indices
    B8 = ds['nbart_nir_1']
    B4 = ds['nbart_red']
    B3 = ds['nbart_green']
    B2 = ds['nbart_blue']
    ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
    ds['NDVI'] = (B8 - B4) / (B8 + B4)

    # Calculate the aggregated metrics
    ds = aggregated_metrics(ds)

    # Remove the temporal bands
    variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
    ds_selected = ds[variables] 

    # Select pixels to use for training/testing
    df = jittered_grid(ds)

    # Change outputs to 0 and 1, instead of 1 and 2
    df['woody_veg'] = df['woody_veg'] - 1

    # Normalize all columns in the DataFrame to be between 0 and 1
    df_normalized = (df - df.min()) / (df.max() - df.min())

    # Save a copy of this dataframe just in case something messes up later (since this is going to take 2 to 4 hours)
    filename = os.path.join(outdir, f"{stub}_df.csv")
    df_normalized.to_csv(filename)
    print("Saved", filename)

    dfs.append(df_normalized)

# Combine all the dataframes
df_all = pd.concat(dfs)

# Should probs save this as a parquet or a feather file instead
filename = os.path.join(outdir, f"woody_veg_preprocessed.csv")
df_all.to_csv(filename)
print("Saved", df_all)