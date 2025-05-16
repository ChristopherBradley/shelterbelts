# +
# Create a training dataset with satellite imagery inputs and tree cover outputs
# -

import os
import glob
import pickle
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage
from scipy.signal import fftconvolve
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback, sys

np.random.seed(0)
random.seed(0)

# +
# Much faster method to calculate spatial variation than scipy.ndimage
def focal_std_fft(array, kernel):
    radius = kernel.shape[0] // 2
    array = np.pad(array, pad_width=radius, mode='reflect')
    
    kernel = kernel / kernel.sum()
    mean = fftconvolve(array, kernel, mode='same')
    mean_sq = fftconvolve(array**2, kernel, mode='same')
    std = np.sqrt(mean_sq - mean**2)
    
    std_unpadded = std[radius:-radius or None, radius:-radius or None]
    return std_unpadded

def make_circular_kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(float)


# -

def aggregated_metrics(ds):
    """Add a temporal median, temporal std, focal mean, and focal std for each temporal band"""
    # Make a list of the variables with a time dimension
    time_vars = [var for var in ds.data_vars if 'time' in ds[var].dims]

    # Calculate aggregated metrics per pixel
    for variable in time_vars:

        # Temporal metrics
        ds_median_temporal = ds[variable].median(dim="time", skipna=True)  # Not sure if I should be doing some kind of outlier removal before this. I should try out the geometric median like Dale Roberts demonstrated.
        ds_std_temporal = ds[variable].std(dim="time", skipna=True)

        # Focal metrics
        radius = 3
        kernel_size = 2 * radius + 1  # 7 pixel diameter because the radius doesn't include the center pixel
        ds_mean_focal_7p = xr.apply_ufunc(
            ndimage.uniform_filter, 
            ds_median_temporal, 
            kwargs={'size': kernel_size, 'mode': 'nearest'}
        )

        kernel = make_circular_kernel(radius=radius)
        std_focal_7p_fft = focal_std_fft(ds_median_temporal.values, kernel)
        ds_std_focal_7p_fft = xr.DataArray(
            std_focal_7p_fft,
            dims=("y", "x")
        )

        ds[f"{variable}_temporal_median"] = ds_median_temporal
        ds[f"{variable}_temporal_std"] = ds_std_temporal
        ds[f"{variable}_focal_mean"] = ds_mean_focal_7p
        ds[f"{variable}_focal_std"] = ds_std_focal_7p_fft

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


def tile_csv_verbose(sentinel_tiles):
    """Run tile_csv and report more info on errors that occur"""
    for sentinel_tile in sentinel_tiles:
        tile_id = "_".join(sentinel_tile.split('/')[-1].split('_')[:2])
        try:
            tile_csv(sentinel_tile)
        except Exception as e:
            print(f"Error in worker {tile_id}:", flush=True)
            traceback.print_exc(file=sys.stdout)


def tile_csv(sentinel_tile):
    """Create a csv file with a subset of training pixels for this tile"""
    
    # I'm currently undecided whether to use a jittered grid or random sample of points. 
    tile_id = "_".join(sentinel_tile.split('/')[-1].split('_')[:2])
    tree_cover_filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{tile_id}_binary_tree_cover_10m.tiff'

    # Load the sentinel imagery and tree cover into an xarray
    with open(sentinel_tile, 'rb') as file:
        ds = pickle.load(file)

    # Load the woody veg and add to the main xarray
    ds1 = rxr.open_rasterio(tree_cover_filename)
    
    # Skip this file if the size is less than 900mx900m
    left, bottom, right, top = ds1.rio.bounds()
    width_m = right - left
    height_m = top - bottom
    if width_m < 900 or height_m < 900:
        print("Tile smaller than 1kmx1km:", tile_id)
        return None
    
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
    df["tile_id"] = tile_id

    # Leaving normalisation for later to help with debugging if I want to visually inspect the raw values
    # df_normalized = (df - df.min()) / (df.max() - df.min())

    # Save a copy of this dataframe just in case something messes up later (since this is going to take 2 to 4 hours)
    filename = os.path.join(outdir, f"{tile_id}_df_tree_cover.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")
    
    return df


def visualise_sample_coords(sentinel_tile="/scratch/xe2/cb8590/Nick_csv/g1_05079_df_tree_cover.csv"):
    """I used this function to visualise the jittered coordinates chosen in QGIS"""
    tile_id = "_".join(sentinel_tile.split('/')[-1].split('_')[:2])
    tree_cover_filename = os.path.join(tree_cover_dir, f"{tile_id}_binary_tree_cover_10m.tiff")
    ds = rxr.open_rasterio(tree_cover_filename)
    crs = ds.rio.crs
    df = tile_csv(sentinel_tile)
    df = df[['tree_cover', 'y', 'x']]

    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['x'], df['y']),
        crs=crs
    )

    # Create transform
    resolution = 10  # meters
    minx, miny, maxx, maxy = gdf.total_bounds
    minx -= resolution / 2
    miny -= resolution / 2
    maxx += resolution / 2
    maxy += resolution / 2
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    # Map points to raster
    tree_cover_array = np.full((height, width), np.nan, dtype=np.float32)
    for idx, row in gdf.iterrows():
        col = int((row.geometry.x - minx) / resolution)
        row_ = int((maxy - row.geometry.y) / resolution)
        if 0 <= row_ < height and 0 <= col < width:
            tree_cover_array[row_, col] = row['tree_cover']

    # Write to GeoTIFF
    filename = f'/scratch/xe2/cb8590/tmp/{tile_id}_training_sample.tif'
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(tree_cover_array, 1)
    print("Saved", filename)


# +
if __name__ == '__main__':
    
    # Load a list of all the downloaded sentinel tiles and training csv's
    tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
    sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
    outdir = "/scratch/xe2/cb8590/Nick_csv3"
    outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"

    sentinel_tiles = glob.glob(f'{sentinel_dir}/*')
    print("num sentinel tiles:", len(sentinel_tiles))

    # csv_tiles = glob.glob(f'{outdir}/*')
    # print("num csv tiles:", len(csv_tiles))

    # +
    # Remove sentinel tiles we've already downloaded
#     sentinel_ids = ["_".join(sentinel_tile.split('/')[-1].split('_')[:2]) for sentinel_tile in sentinel_tiles]
#     csv_ids = ["_".join(csv_tile.split('/')[-1].split('_')[:2]) for csv_tile in csv_tiles]

#     is_news = [(sentinel_id not in csv_ids) for sentinel_id in sentinel_ids]
#     sentinel_tiles = [sentinel_tile for sentinel_tile, is_new in zip(sentinel_tiles, is_news) if is_new]
#     print("num sentinel tiles not yet downloaded: ", len(sentinel_tiles))
#     # -

    # Randomise the tiles so I can have a random sample before they all complete
    sentinel_randomised = random.sample(sentinel_tiles, len(sentinel_tiles))

    # rows = sentinel_randomised[:16]
    # workers = 4
    rows = sentinel_randomised
    workers = 50
    batch_size = math.ceil(len(rows) / workers)
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    print("num_batches: ", len(batches))
    print("num tiles in first batch", len(batches[0]))

    # +
    # %%time
    with ProcessPoolExecutor(max_workers=workers) as executor:
        print(f"Starting {workers} workers, with {batch_size} rows each")
        futures = [executor.submit(tile_csv_verbose, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with: {e}", flush=True)

    # 60 secs for 16 tiles x 2 batches on Large compute (7 cores)
    # Took 1 hour 33 mins to do all 7791 tiles with XLarge computer, with batches of 50 tiles (14 cores)

    # +
    # Create a dataframe of imagery and tree cover classifications for each tile
    csv_tiles = glob.glob(f'{outdir}/*')
    print("num csv tiles now:", len(csv_tiles))  # Why did 11 of the pickle files not get converted to csv files?
    # -

    dfs = []
    for csv_tile in csv_tiles:
        df = pd.read_csv(csv_tile, index_col=False)
        dfs.append(df)

    # Combine all the dataframes
    df_all = pd.concat(dfs)

    # %%time
    # Feather file is more efficient, but csv is more readable. Anything over 100MB I should probs use a feather file.
    filename = os.path.join(outlines_dir, f"tree_cover_preprocessed3.feather")
    df_all.to_feather(filename)
    print("Saved", filename)
