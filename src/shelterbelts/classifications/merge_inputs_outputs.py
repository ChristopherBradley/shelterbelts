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
    
    var = mean_sq - mean**2
    var = np.maximum(var, 0)   # prevent negative due to round-off
    std = np.sqrt(var)

    # std = np.sqrt(mean_sq - mean**2)
    
    std_unpadded = std[radius:-radius or None, radius:-radius or None]
    return std_unpadded

def make_circular_kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(float)


# -

# I kinda want to add the other aggregated metrics option of 12 separate months. 
def aggregated_metrics(ds, radius=4):
    """Add a temporal median, temporal std, focal mean, and focal std for each temporal band"""
    # Make a list of the variables with a time dimension
    time_vars = [var for var in ds.data_vars if 'time' in ds[var].dims]

    # Calculate aggregated metrics per pixel
    for variable in time_vars:

        # Temporal metrics
        ds_median_temporal = ds[variable].median(dim="time", skipna=True)  # Not sure if I should be doing some kind of outlier removal before this. I should try out the geometric median like Dale Roberts demonstrated.
        ds_std_temporal = ds[variable].std(dim="time", skipna=True)

        # Focal metrics
        kernel_size = 2 * radius + 1  # 7 pixel diameter because the radius doesn't include the center pixel
        ds_mean_focal_7p = xr.apply_ufunc(
            ndimage.uniform_filter, 
            ds_median_temporal, 
            kwargs={'size': kernel_size, 'mode': 'nearest'},
            dask="parallelized",  # Lazy loading
            output_dtypes=[ds_median_temporal.dtype],
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

def jittered_grid(ds, spacing=10):
    """Create an equally spaced grid with a random jitter"""

    # Calculate grid
    spacing_x = spacing
    spacing_y = spacing
    half_spacing_x = spacing_x // 2
    half_spacing_y = spacing_y // 2

    max_jitter = math.ceil(spacing/2) - 1
    jitter_range = list(range(-max_jitter, max_jitter + 1))

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

    # Much faster way of adding x and y coordinates to the dataframe
    y_coords = yy_jittered_coords.ravel()
    x_coords = xx_jittered_coords.ravel()

    subset = ds[aggregated_vars].interp(y=("points", y_coords), x=("points", x_coords))
    df = subset.to_dataframe().reset_index()
    
    df = df.drop(columns='points')
    
    return df


def visualise_jittered_grid(ds, spacing=10, outdir='/scratch/xe2/cb8590/tmp', stub="TEST"):
    """Save a geopackage so you can visualise the jittered grid in QGIS"""
    df = jittered_grid(ds, spacing)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['x'], df['y']),
        crs=ds.rio.crs
    )
    filename = os.path.join(outdir, f'{stub}_jittered_s{spacing}.gpkg')
    gdf.to_file(filename) 
    print("Saved: ", filename)

def tile_csv_ds(ds, tree_file, outdir, radius=4, spacing=10, verbose=True):
    """Create a csv file with a subset of training pixels for this sentinel xarray"""

    # Load the woody veg
    if verbose:
        print(f"Loading {tree_file}")
    da = rxr.open_rasterio(tree_file).isel(band=0).drop_vars('band')
    

    # Should really make sure there's always a crs in lidar.py before writing out the tif file
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:28355", inplace=True)

    # Match the tree tif. Reprojecting the other way round breaks if a sentinel dimension is larger thna a tree dimension
    ds = ds.rio.reproject_match(da)  
    for var in ds.data_vars:
        ds[var].attrs.pop("grid_mapping", None)

    # Add the woody veg to the main xarray
    ds['tree_cover'] = da.astype(float)

    # Calculate vegetation indices
    B8 = ds['nbart_nir_1']
    B4 = ds['nbart_red']
    B3 = ds['nbart_green']
    B2 = ds['nbart_blue']
    ds['EVI'] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
    ds['NDVI'] = (B8 - B4) / (B8 + B4)
    ds['GRNDVI'] = (B8 - B3 + B4) / (B8 + B3 + B4)

    # Calculate the aggregated metrics
    ds = aggregated_metrics(ds, radius)

    # Remove the temporal bands
    variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
    ds = ds[variables] 

    # I'm currently undecided whether to use a jittered grid or random sample of points. 
    df = jittered_grid(ds, spacing)
    stub = tree_file.split('/')[-1].split('.')[0]  # filename without the path
    df["tile_id"] = stub

    # Save a copy of this dataframe just in case something messes up later
    filename = os.path.join(outdir, f"{stub}_df_r{radius}_s{spacing}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

    del ds 
    del da
    gc.collect()
    
    return df


def tile_csv(sentinel_file, tree_file, outdir, radius=4, spacing=10, verbose=True):
    """Create a csv file with a subset of training pixels for this sentinel filename"""
    with open(sentinel_file, 'rb') as file:
        if verbose:
            print(f"Loading {sentinel_file}")
        ds = pickle.load(file)
    df = tile_csv_ds(ds, tree_file, outdir, radius, spacing, verbose)
    return df


def tile_csvs(sentinel_folder, tree_folder, outdir=".", radius=4, spacing=10, limit=None, double_f=False, specific_usecase=None):
    """Run tile_csv on all the tiles and report info on any errors that occur"""
    
    # Randomise the tiles so I can have a random sample before they all complete
    sentinel_tiles = glob.glob(f'{sentinel_folder}/*')

    if specific_usecase is not None:
        footprints_percent = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg'
        gdf_percent = gpd.read_file(footprints_percent)
        footprints_years = '/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg'
        gdf_year = gpd.read_file(footprints_years)
        gdf = gdf_percent.merge(gdf_year[['filename', 'year']])
        gdf['stub'] = [stub.split(".")[0] for stub in gdf['filename']]
        gdf_recent = gdf[~gdf['bad_tif'] & (gdf['year'] > 2017)] 
        
    if specific_usecase == 'lidar_year':
        # get just the sentinel_tiles matching the year of the lidar acquisition
        sentinel_tiles = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year']}_ds2_{row['year']}.pkl" for i, row in gdf_recent.iterrows()]
        assert(all(os.path.exists(f) for f in sentinel_tiles))
    if specific_usecase == 'previous_year':
        # Imagery from the year before lidar was taken
        sentinel_tiles = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year'] - 1}_ds2_{row['year'] - 1}.pkl" for i, row in gdf_recent.iterrows()]
        assert(all(os.path.exists(f) for f in sentinel_tiles))
    if specific_usecase == 'next_year':
        # Imagery from the year before lidar was taken
        sentinel_tiles = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year'] + 1}_ds2_{row['year'] + 1}.pkl" for i, row in gdf_recent.iterrows()]
        assert(all(os.path.exists(f) for f in sentinel_tiles))
    if specific_usecase == 'previous_2_years':
        # Imagery from the current and previous years
        sentinel_filenames = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year']}_ds2_{row['year']}.pkl" for i, row in gdf_recent.iterrows()]
        sentinel_filenames2 = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year'] + 1}_ds2_{row['year'] + 1}.pkl" for i, row in gdf_recent.iterrows()]
        sentinel_tiles = sentinel_filenames + sentinel_filenames2
    if specific_usecase == 'next_2_years':
        # Imagery from the current and next year
        sentinel_filenames = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year']}_ds2_{row['year']}.pkl" for i, row in gdf_recent.iterrows()]
        sentinel_filenames2 = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year'] + 1}_ds2_{row['year'] + 1}.pkl" for i, row in gdf_recent.iterrows()]
        sentinel_tiles = sentinel_filenames + sentinel_filenames2
    if specific_usecase == 'all_years':
        # Imagery from all years 2017-2024
        sentinel_tiles = glob.glob(f'{sentinel_folder}/*') # I could also try including the 'bad' tiles in /scratch/xe2/cb8590/Nick_sentinel2

    sentinel_randomised = random.sample(sentinel_tiles, len(sentinel_tiles))
    if limit is not None:
        sentinel_randomised = sentinel_randomised[:limit]
        
    print("About to process n tiles:", len(sentinel_tiles))
    for sentinel_tile in sentinel_randomised:
        stub = "_".join(sentinel_tile.split('/')[-1].split('_')[:-2])   # Remove the _ds2_year
        tree_file = os.path.join(tree_folder, f"{stub}.tif{'f' if double_f else ''}")  # Should probably just rename all the files to have the same suffix instead
        tile_csv(sentinel_tile, tree_file, outdir, radius, spacing)

# +
# sentinel_folder = '/scratch/xe2/cb8590/Nick_sentinel'
# tree_folder = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
# sentinel_tiles = glob.glob(f'{sentinel_folder}/*')

# footprints_percent = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg'
# gdf_percent = gpd.read_file(footprints_percent)
# footprints_years = '/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg'
# gdf_year = gpd.read_file(footprints_years)
# gdf = gdf_percent.merge(gdf_year[['filename', 'year']])
# gdf['stub'] = [stub.split(".")[0] for stub in gdf['filename']]

# gdf_recent = gdf[~gdf['bad_tif'] & (gdf['year'] > 2017)] 

# # The easiest way will be to just construct all the sentinel files from the gdf_recent. Then maybe just assert they all exist.
# sentinel_filenames = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year']}_ds2_{row['year']}.pkl" for i, row in gdf_recent.iterrows()]
# sentinel_filenames2 = [f"/scratch/xe2/cb8590/Nick_sentinel/{row['stub']}_{row['year'] + 1}_ds2_{row['year'] + 1}.pkl" for i, row in gdf_recent.iterrows()]
# sentinel_tiles = sentinel_filenames + sentinel_filenames2

# all(os.path.exists(f) for f in sentinel_filenames)


# +
# Haven't yet added this into the main preprocess pipeline
# def attach_koppen_classes(df):
#     """Attach the koppen class of each tile to the relevant rows"""
#     # This should really all happen in merge_inputs_outputs
#     df = df[(df['tree_cover'] == 0) | (df['tree_cover'] == 1)]     # Drop the 174 rows where tree cover values = 2
#     df = df[df.notna().all(axis=1)]     # Drop the two rows with NaN values

#     # Add the bioregion to the training/testing data
#     gdf = gpd.read_file(filename_centroids)

#     gdf['tile_id'] = ["_".join(filename.split('/')[-1].split('_')[:2]) for filename in gdf['filename']]
#     gdf = gdf.rename(columns={'Full Name':'koppen_class'})
#     df = df.merge(gdf[['tile_id', 'koppen_class']])

#     # Should also use the y, x, tile_id to get a coord in the Australia EPSG:7844

#     return df
# filename_centroids = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/centroids_named.gpkg"
# df = attach_koppen_classes(df)

# -

def preprocess(sentinel_folder, tree_folder=None, outdir=".", stub="TEST", radius=4, spacing=10, outlines_gpkg=None, limit=None, specific_usecase=None):
    """Merge the inputs and outputs for training the model
    
    Parameters
    ----------
        sentinel_folder: Folder with sentinel pickle files generated by sentinel.py
        tree_folder: Folder with binary tree tifs generated by lidar.py or util/binary_trees.py 
            - if not provided, then the sentinel_folder gets processed as normal, just without any ground truth data
        outdir: Folder for the output feather file
        stub: prefix of feather file
        radius: The size of the kernel used for spatial variation
        spacing: The distance between jittered points
        outlines_gpkg: Extra metadata such as koppen_class generated by koppen.py. Should contain at least a column 'filename' corresponding to the sentinel_filenames
            - if not provided, then we just don't attach this extra metadata
        limit: The number of files to process
            - if none, then run on all the files in the sentinel_folder
            
    Returns
    -------
        gdf: Geodataframe with the inputs and outputs for each tile
    
    Downloads
    ---------
        feather: Pandas feather file (database) of the gdf
    
    """
    # Create a csv of inputs and outputs for each tile
    tile_csvs(sentinel_folder, tree_folder, outdir, radius, spacing, limit, specific_usecase)

    # Merge the results into a single feather file. 
    csv_tiles = glob.glob(os.path.join(outdir, '*.csv'))
    dfs = []
    for csv_tile in csv_tiles:
        df = pd.read_csv(csv_tile, index_col=False)
        dfs.append(df)
    df_all = pd.concat(dfs)
    filename = os.path.join(outdir, f"{stub}_preprocessed.feather")
    df_all.to_feather(filename)
    print("Saved", filename)
                          
    return df_all


# +
import argparse

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sentinel_folder', required=True, help='Folder with sentinel pickle files generated by sentinel.py')
    parser.add_argument('--tree_folder', default=None, help='Folder with binary tree tifs generated by lidar.py or util/binary_trees.py (optional)')
    parser.add_argument('--outdir', default='.', help='Folder for the output feather file (default: current directory)')
    parser.add_argument('--stub', default='TEST', help='Folder for the output feather file (default: current directory)')
    parser.add_argument('--radius', type=int, default=4, help='Size of the kernel used for spatial variation (default: 4)')
    parser.add_argument('--spacing', type=int, default=10, help='Distance between jittered points (default: 10)')
    parser.add_argument('--outlines_gpkg', default=None, help='Optional GPKG with metadata (should contain a "filename" column)')
    parser.add_argument('--limit', type=int, default=None, help='Number of files to process (default: all)')
    parser.add_argument('--specific_usecase', default=None, help='Specific scenarios for trying out different years with Nicks training data)

    return parser.parse_args()


# # %%time
if __name__ == '__main__':
    args = parse_arguments()
    
    preprocess(
        sentinel_folder=args.sentinel_folder,
        tree_folder=args.tree_folder,
        outdir=args.outdir,
        stub=args.stub,
        radius=args.radius,
        spacing=args.spacing,
        outlines_gpkg=args.outlines_gpkg,
        limit=args.limit,
        specific_usecase=args.specific_usecase
    )


# +
# # %%time
# outdir = '../../../outdir'
# tree_folder = '../../../data'
# sentinel_tile = os.path.join(outdir,'g2_26729_binary_tree_cover_10m.pkl ')
# tile_csv(sentinel_tile, tree_folder, outdir, double_f=True)

# +
# sentinel_folder = "/scratch/xe2/cb8590/Tas_sentinel"
# tree_folder = "/scratch/xe2/cb8590/Tas_tifs"
# outdir = f"/scratch/xe2/cb8590/Tas_csv"
# preprocess(sentinel_folder, tree_folder, outdir, limit=1)

# +
# # %%time
# data_dir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
# stub = 'g2_26729_binary_tree_cover_10m'
# outdir = '/scratch/xe2/cb8590/tmp'
# verbose = True
# radius = 4
# spacing = 5
# double_f = True

# tree_file = f'{data_dir}/{stub}.tiff'
# sentinel_file = f'{outdir}/{stub}_ds2_2020.pkl'
# # tile_csv(sentinel_file, tree_file=tree_file, outdir=outdir, radius=4, spacing=1)

# +
# sentinel_file = '/scratch/xe2/cb8590/Tas_sentinel/NorthernMidlands2019-C1-AHD_5375370_GDA2020_55_woodyveg_res10_cat5_ds2_2019.pkl'
# tree_file =  '/scratch/xe2/cb8590/Tas_tifs/NorthernMidlands2019-C1-AHD_5375370_GDA2020_55_woodyveg_res10_cat5.tif'
# with open(sentinel_file, 'rb') as file:
#     ds2 = pickle.load(file)
# da = rxr.open_rasterio(tree_file).isel(band=0).drop_vars('band')
# ds['nbart_red'].isel(time=0).rio.to_raster('/scratch/xe2/cb8590/TAS_red0.tif')
