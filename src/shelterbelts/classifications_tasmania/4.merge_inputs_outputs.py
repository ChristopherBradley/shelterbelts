# +
# Create a training dataset with satellite imagery inputs and woody veg or canopy cover outputs
# -

import os
import glob
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage

# +
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 100)
# -

outdir = "/g/data/xe2/cb8590/shelterbelts/"

tiles = glob.glob("/g/data/xe2/cb8590/shelterbelts/*_ds2.pkl")
sub_stub = 'canopycover'

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
    """Create an equally spaced 10x10 coordinate grid with a random 2 pixel jitter"""

    # Calculate grid
    spacing_x = 10
    spacing_y = 10
    half_spacing_x = spacing_x // 2
    half_spacing_y = spacing_y // 2
    jitter_range = [-2, -1, 0, 1, 2]

    # Regular grid
    y_inds = np.arange(half_spacing_y, ds.sizes['y'] - half_spacing_y, spacing_y)
    x_inds = np.arange(half_spacing_x, ds.sizes['x'] - half_spacing_x, spacing_x)

    # Create full grid
    yy, xx = np.meshgrid(y_inds, x_inds, indexing='ij')

    # Apply random jitter to each (row, col) separately
    yy_jittered = yy + np.random.choice(jitter_range, size=yy.shape)
    xx_jittered = xx + np.random.choice(jitter_range, size=xx.shape)

    # Get actual coordinates. Note that these coords are in the EPSG of the original raster.
    yy_jittered_coords = ds['y'].values[yy_jittered]
    xx_jittered_coords = ds['x'].values[xx_jittered]

    # Get non-time-dependent variables
    aggregated_vars = [var for var in ds.data_vars if 'time' not in ds[var].dims]

    data_list = []
    for y_coord, x_coord in zip(yy_jittered_coords.ravel(), xx_jittered_coords.ravel()):
        values = {var: ds[var].sel(y=y_coord, x=x_coord, method='nearest').item() for var in aggregated_vars}
        values.update({'y': y_coord, 'x': x_coord})
        data_list.append(values)

    # Create DataFrame
    df = pd.DataFrame(data_list)
    return df


# %%time
# Used this code to visualise the jittered coordinates in QGIS
def visualise_jittered_gdf():
    gdfs = []
    for tile in tiles:
        # tile = tiles[2]
        print(tile)
        stub = tile.replace(outdir,"").replace("_ds2.pkl","")
        filename = os.path.join(outdir, f"{stub}_{sub_stub}_2019.tif")
        da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
        ds = xr.Dataset({
                "canopy_cover": da
            })
        df = jittered_grid(ds)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['x'], df['y']),
            crs=ds.rio.crs
        )
        gdfs.append(gdf)

    gdf_all = pd.concat(gdfs)
    filename = f'/scratch/xe2/cb8590/tmp/all_tasmania_jitter.gpkg'
    gdf_all.to_file(filename) # Save as geopackage
    print(filename)


# +
# %%time
# Create a dataframe of imagery and woody veg or canopy cover classifications for each tile
# sub_stub = 'woodyveg'
dfs = []
for tile in tiles:
    stub = tile.replace(outdir,"").replace("_ds2.pkl","")

    # Load the imagery
    with open(tile, 'rb') as file:
        ds = pickle.load(file)

    # Load the woody veg and add to the main xarray
    filename = os.path.join(outdir, f"{stub}_{sub_stub}_2019.tif")
    ds1 = rxr.open_rasterio(filename)
    ds[sub_stub] = ds1.isel(band=0).drop_vars('band')

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
    df[sub_stub] = df[sub_stub] - 1

    # Normalize all columns in the DataFrame to be between 0 and 1
    df_normalized = (df - df.min()) / (df.max() - df.min())

    # Save a copy of this dataframe just in case something messes up later (since this is going to take 2 to 4 hours)
    filename = os.path.join(outdir, f"{stub}_df_{sub_stub}.csv")
    df_normalized.to_csv(filename)
    print("Saved", filename)

    dfs.append(df_normalized)

# 1 min 30 secs each 
# -

# Combine all the dataframes
df_all = pd.concat(dfs)

# Should probs save this as a parquet or a feather file instead
filename = os.path.join(outdir, f"{sub_stub}_preprocessed.csv")
df_all.to_csv(filename)
print("Saved", df_all)


