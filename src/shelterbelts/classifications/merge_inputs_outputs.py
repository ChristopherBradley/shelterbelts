import os
import sys
import glob
import pickle
import math
import random
import gc
import shutil
import argparse

import numpy as np
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import scipy.ndimage as ndimage
from scipy.signal import fftconvolve

from shelterbelts.utils.filepaths import tmpdir

# Monkey fix to load the new pickle files in an older version of numpy
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.numeric'] = np.core.numeric

np.random.seed(0)
random.seed(0)

def focal_std_fft(array, kernel):
    """FFT-based rolling standard deviation (much faster than my old version using scipy.ndimage)."""
    radius = kernel.shape[0] // 2
    array = np.pad(array, pad_width=radius, mode='reflect')

    kernel = kernel / kernel.sum()
    mean = fftconvolve(array, kernel, mode='same')
    mean_sq = fftconvolve(array**2, kernel, mode='same')

    var = mean_sq - mean**2
    var = np.maximum(var, 0)   # prevent negative due to round-off
    std = np.sqrt(var)

    std_unpadded = std[radius:-radius or None, radius:-radius or None]
    return std_unpadded


def make_circular_kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(float)


def aggregated_metrics(ds, radius=4):
    """Add a temporal median, temporal std, focal mean, and focal std for each temporal band"""
    # Make a list of the variables with a time dimension
    time_vars = [var for var in ds.data_vars if 'time' in ds[var].dims]

    # Calculate aggregated metrics per pixel
    for variable in time_vars:

        # Temporal metrics
        ds_median_temporal = ds[variable].median(dim="time", skipna=True)
        ds_std_temporal = ds[variable].std(dim="time", skipna=True)

        # Focal metrics
        kernel_size = 2 * radius + 1  # 7 pixel diameter because the radius doesn't include the centre pixel
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
    """Sample pixels on a regular spacing grid with per-point random jitter, returned as a long-format dataframe."""

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


def visualise_jittered_grid(ds, spacing=10, outdir=tmpdir, stub="TEST"):
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


def merge_inputs_outputs_ds(ds, tree_file, outdir, radius=4, spacing=10, verbose=False, alpha_folder=None):
    """Create a training CSV from an in-memory Sentinel-2 Dataset and a tree-cover tif.

    Parameters
    ----------
    ds : xarray.Dataset
        Sentinel-2 Dataset with 10 nbart_* bands on a time/y/x grid.
    tree_file : str
        Path to a binary tree-cover GeoTIFF (0 = non-tree, 1 = tree).
    outdir : str
        Output directory for saving the CSV.
    radius : int, optional
        Radius (in pixels) of the circular kernel used for focal mean/std features.
    spacing : int, optional
        Sampling grid spacing in pixels. A larger spacing produces fewer,
        less-correlated rows.
    verbose : bool, optional
        Print progress messages.
    alpha_folder : str, optional
        Folder containing AlphaEarth embedding pickles to append as extra features.

    Returns
    -------
    pandas.DataFrame
        The training rows, also written to
        ``{outdir}/{stub}_df_r{radius}_s{spacing}_{year}.csv``.
    """
    if verbose:
        print(f"Loading {tree_file}")
    da = rxr.open_rasterio(tree_file).isel(band=0).drop_vars('band')

    # Should really make sure there's always a crs in lidar.py before writing out the tif file
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:28355", inplace=True)

    # Match the tree tif. Reprojecting the other way round breaks if a sentinel dimension is larger than a tree dimension
    try:
        ds = ds.rio.reproject_match(da)
    except:
        return None  # Seems like sometimes either the ds or da is empty, giving this error: rasterio._err.CPLE_IllegalArgError: GDALWarpOptions.Validate(): nBandCount=0, no bands configured!

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

    year = str(ds.time[0].dt.year.values)
    start_date = str(ds.time[0].dt.date.item())
    end_date = str(ds.time[-1].dt.date.item())

    if alpha_folder is not None:
        stub = tree_file.split('/')[-1].split('.')[0]
        alpha_file = os.path.join(alpha_folder, f'{stub}_alpha_earth_embeddings_{year}.pkl')
        with open(alpha_file, 'rb') as file:
            ds_alpha = pickle.load(file)
        ds_alpha = ds_alpha.rio.reproject_match(ds)
        for i in range(ds_alpha.sizes['band']):
            ds[f'alpha_embedding_{i+1}'] = ds_alpha.isel(band=i)

    # Remove the temporal bands
    variables = [var for var in ds.data_vars if 'time' not in ds[var].dims]
    ds = ds[variables]

    df = jittered_grid(ds, spacing)
    stub = tree_file.split('/')[-1].split('.')[0]  # filename without the path
    df["tile_id"] = stub
    df["year"] = year
    df["start_date"] = start_date
    df['end_date'] = end_date

    df = df.drop(columns=['spatial_ref', 'band'], errors='ignore')

    # Change the datatypes to make the file as small as possible
    df = df.drop(columns=['start_date', 'end_date'], errors='ignore')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int16)

    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"{stub}_df_r{radius}_s{spacing}_{year}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

    del ds
    del da
    gc.collect()

    return df


def merge_inputs_outputs(sentinel_pickle, tree_tif, outdir=".", radius=4, spacing=10, verbose=False):
    """
    Build a per-pixel training CSV from a Sentinel-2 pickle and a tree-cover tif.

    Each row of the CSV is one sampled pixel; columns are the temporally-aggregated
    Sentinel features plus the binary tree label drawn from tree_tif. The resulting
    CSV feeds directly into
    :func:`shelterbelts.classifications.neural_network.train_neural_network`.

    Parameters
    ----------
    sentinel_pickle : str
        Path to a Sentinel-2 xarray.Dataset pickle with 10 nbart_* bands on a
        time/y/x grid. Typically produced by
        :func:`shelterbelts.classifications.sentinel_nci.download_ds2_bbox`.
    tree_tif : str
        Path to a binary tree-cover GeoTIFF (0 = non-tree, 1 = tree).
    outdir : str, optional
        Output directory for saving results.
    radius : int, optional
        Radius (in pixels) of the circular kernel used for focal mean/std features.
    spacing : int, optional
        Sampling grid spacing in pixels. A larger spacing produces fewer,
        less-correlated rows.
    verbose : bool, optional
        Print progress messages.

    Returns
    -------
    pandas.DataFrame
        The training rows, also written to
        ``{outdir}/{stub}_df_r{radius}_s{spacing}_{year}.csv``.
    """
    with open(sentinel_pickle, 'rb') as file:
        if verbose:
            print(f"Loading {sentinel_pickle}")
        ds = pickle.load(file)
    return merge_inputs_outputs_ds(ds, tree_tif, outdir, radius, spacing, verbose)


def tile_csvs(sentinel_folder, tree_folder, outdir=".", radius=4, spacing=10, limit=None, double_f=False, specific_usecase=None):
    """
    Run :func:`merge_inputs_outputs` on every Sentinel pickle in sentinel_folder.

    Parameters
    ----------
    sentinel_folder : str
        Folder containing Sentinel-2 pickle files produced by sentinel_nci.py or
        sentinel_dea.py.
    tree_folder : str
        Folder containing binary tree-cover tifs. Each tif is matched to a pickle
        by stem name.
    outdir : str, optional
        See :func:`merge_inputs_outputs`.
    radius : int, optional
        See :func:`merge_inputs_outputs`.
    spacing : int, optional
        See :func:`merge_inputs_outputs`.
    limit : int, optional
        Maximum number of tiles to process. If None, all tiles are processed.
    double_f : bool, optional
        Use ``.tiff`` extension for tree files instead of ``.tif``.
    """
    sentinel_tiles = glob.glob(f'{sentinel_folder}/*')
    sentinel_randomised = random.sample(sentinel_tiles, len(sentinel_tiles))
    if limit is not None:
        sentinel_randomised = sentinel_randomised[:limit]

    print("About to process n tiles:", len(sentinel_randomised))
    for sentinel_tile in sentinel_randomised:
        stub = "_".join(sentinel_tile.split('/')[-1].split('_')[:-3])   # Remove the year_ds2_year
        tree_file = os.path.join(tree_folder, f"{stub}.tif{'f' if double_f else ''}")
        merge_inputs_outputs(sentinel_tile, tree_file, outdir, radius, spacing)


def tiles_todo(sentinel_folder, csv_folder):
    """Move Sentinel pickles that don't yet have a matching training CSV into a tiles_todo subfolder for re-processing."""

    sentinel_folder = '/scratch/xe2/cb8590/Nick_sentinel/*.pkl'
    csv_folder = '/scratch/xe2/cb8590/Nick_training_lidar_year/*.csv'
    sentinel_files = glob.glob(sentinel_folder)
    csv_files = glob.glob(csv_folder)

    sentinel_stubs = ['_'.join(sentinel_file.split('/')[-1].split('_')[:2]) for sentinel_file in sentinel_files]
    sentinel_years = [sentinel_file.split('.')[0][-4:] for sentinel_file in sentinel_files]
    csv_stubs = ['_'.join(csv_file.split('/')[-1].split('_')[:2]) for csv_file in csv_files]
    csv_years = [csv_file.split('.')[0][-4:] for csv_file in csv_files]

    sentinel_pairs = set(zip(sentinel_stubs, sentinel_years))
    csv_pairs = set(zip(csv_stubs, csv_years))
    missing_pairs = sentinel_pairs - csv_pairs
    missing_files = [f'/scratch/xe2/cb8590/Nick_sentinel/{missing_pair[0]}_binary_tree_cover_10m_{missing_pair[1]}_ds2_{missing_pair[1]}.pkl' for missing_pair in missing_pairs]
    src_dir = '/scratch/xe2/cb8590/Nick_sentinel'
    dst_dir = os.path.join(src_dir, 'tiles_todo')
    os.makedirs(dst_dir, exist_ok=True)
    for file in missing_files:
        filename = os.path.basename(file)
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"File not found: {src_path}")


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('sentinel_folder', help='Folder with sentinel pickle files generated by sentinel_nci.py or sentinel_dea.py')
    parser.add_argument('--tree_folder', default=None, help='Folder with binary tree tifs generated by lidar.py or binary_trees.py')
    parser.add_argument('--outdir', default='.', help='Output directory for CSV files (default: current directory)')
    parser.add_argument('--radius', type=int, default=4, help='Focal kernel radius in pixels (default: 4)')
    parser.add_argument('--spacing', type=int, default=10, help='Jittered grid spacing in pixels (default: 10)')
    parser.add_argument('--limit', type=int, default=None, help='Number of tiles to process (default: all)')
    parser.add_argument('--double_f', action="store_true", help='Use .tiff extension instead of .tif for tree files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    tile_csvs(
        sentinel_folder=args.sentinel_folder,
        tree_folder=args.tree_folder,
        outdir=args.outdir,
        radius=args.radius,
        spacing=args.spacing,
        limit=args.limit,
        double_f=args.double_f,
    )
