import os
import glob
import argparse
import math
import pathlib

import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box

# Trying to avoid memory issues
import gc
import psutil
import subprocess, sys

from shelterbelts.utils.tiles import merge_tiles_bbox, merged_ds, crop_and_rasterize
from shelterbelts.apis.barra_daily import barra_daily

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import patch_metrics

# 11 secs for all these imports
# -
from shelterbelts.utils.filepaths import (
    default_outdir,
    default_tmpdir,
    worldcover_dir,
    worldcover_geojson,
    hydrolines_gdb,
    roads_gdb,
)

process = psutil.Process(os.getpid())


GEE_legend = {
  0: 'Not Trees',
  10: 'Tree cover',
  11: 'Scattered Trees',
  12: 'Patch Core',
  13: 'Patch Edge',
  14: 'Other Trees',
  15: 'Trees in Gullies',
  16: 'Trees on Ridges',
  17: 'Trees next to Roads',
  18: 'Linear Patches',
  19: 'Non-linear Patches',
  20: 'Shrubland',
  30: 'Grassland',
  31: 'Unsheltered Grassland',
  32: 'Sheltered Grassland',
  40: 'Cropland',
  41: 'Unsheltered Cropland',
  42: 'Sheltered Cropland',
  50: 'Built-up',
  60: 'Bare',
  70: 'Snow and ice',
  80: 'Permanent water bodies',
  90: 'Herbaceous wetland',
  95: 'Mangroves',
  100: 'Moss and lichen'
}

def indices_tif(percent_tif, outdir=default_outdir,
                     tmpdir=default_tmpdir, stub=None,
                     wind_method=None, wind_threshold=20,
                     cover_threshold=1, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                     crop_pixels=0, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6,
                     debug=False):
    """
    Run the complete indices pipeline for a single percent-cover GeoTIFF. 

    Parameters
    ----------
    percent_tif : str
        Path to the input percent-cover GeoTIFF (one-band, percent tree cover). 
    outdir : str, optional
        Output directory for saving results (default is in utils.filepaths).
    tmpdir : str, optional
        Directory for temporary files (default is in utils.filepaths).
    stub : str, optional
        Prefix for output filenames. If not provided it is derived from ``percent_tif``.
    wind_method : str or None, optional
        Method used to infer shelter direction. Can be ``None``,
        ``'WINDWARD'``, ``'MOST_COMMON'``, ``'MAX'``, ``'HAPPENED'`` or
        ``'ANY'``. See :func:`shelter_categories` for details.
    wind_threshold : int, optional
        Wind speed threshold in km/h. Default 20.
    cover_threshold : int, optional
        Pixel percent cover threshold to treat a pixel as 'tree'. Default 1.
        - If input is a binary tif use ``cover_threshold=1``.
        - For percent-cover tifs typical values are 10 or 20.
        - For confidence tifs a value like 50 is reasonable.
    min_patch_size : int, optional
        Minimum area (pixels) to classify as a patch rather than scattered trees.
        Default is 20.
    edge_size : int, optional
        Distance (pixels) defining the edge region around patch cores.
        Default is 3.
    max_gap_size : int, optional
        Maximum gap (pixels) to bridge when connecting tree clusters.
        Default is 1.
    distance_threshold : int, optional
        Distance from trees that counts as sheltered.
        Units are either 'tree heights' or 'number of pixels', depending on if a height_tif is provided.
        Default is 20.
    density_threshold : int, optional
        Percentage tree cover within the ``distance_threshold`` that counts as sheltered.
        Only applies if the ``wind_data`` is not provided. Default is 5.
    buffer_width : int, optional
        Number of pixels away from the feature that still counts as within the buffer. Default is 3.
    strict_core_area : bool, optional
        If True, enforce that core areas exceed the edge_size at all points.
        If False, use dilation and erosion to allow some irregularity. Default is True.
    crop_pixels : int, optional
        Number of pixels to crop from each edge of the output. Default is 0.
    min_core_size : int, optional
        Minimum area (pixels) to classify as a core area. Default is 1000.
    min_shelterbelt_length : int, optional
        Minimum skeleton length (in pixels) to classify a cluster as linear. Default is 15.
    max_shelterbelt_width : int, optional
        Maximum skeleton width (in pixels) to classify a cluster as linear. Default is 6.
    debug : bool, optional
        If True, intermediate TIFFs/plots are saved for debugging. Default False.

    """
    if stub is None:
        # stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'
        stub = percent_tif.split('/')[-1].split('.')[0][:50] # Hopefully there's something unique in the first 50 characters
    # Extract data_folder from ELVIS filenaming system, or use a generic stub if not found
    data_folder_idx = percent_tif.find('DATA')
    if data_folder_idx != -1:
        data_folder = percent_tif[data_folder_idx:data_folder_idx + 11]
    else:
        data_folder = 'generic'

    da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
    da_trees = da_percent >= cover_threshold

    gs_bounds = gpd.GeoSeries([box(*da_percent.rio.bounds())], crs=da_percent.rio.crs)
    bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
    
    # import pdb; pdb.set_trace()
    # Anything that might be run in parallel needs a unique filename, so we don't get rasterio merge conflicts
    worldcover_stub = f'{data_folder}_{stub}_{wind_method}_w{wind_threshold}_c{cover_threshold}_m{min_patch_size}_e{edge_size}_g{max_gap_size}_di{distance_threshold}_de{density_threshold}_b{buffer_width}_mc{min_core_size}_msl{min_shelterbelt_length}_msw{max_shelterbelt_width}_sca{strict_core_area}' # 
    
    mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, worldcover_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    gdf_hydrolines, ds_hydrolines = crop_and_rasterize(da_percent, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, feature_name='gullies')
    gdf_roads, ds_roads = crop_and_rasterize(da_percent, roads_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, layer='NationalRoads_2025_09', feature_name='roads')

    if wind_method and wind_method != "None":  # Handling conversion of None to "None" when using subprocess
        lat = (bbox_4326[1] + bbox_4326[3])/2
        lon = (bbox_4326[0] + bbox_4326[2])/2
        ds_wind = barra_daily(lat=lat, lon=lon, start_year=2020, end_year=2020, gdata=True, plot=False, save_netcdf=False) # This line is currently the limiting factor since it takes 4 secs
    else:
        # if no wind_method provided than the percent_cover method without wind gets used
        ds_wind = None

    ds_woody_veg = da_trees.to_dataset(name='woody_veg')
    ds_tree_categories = tree_categories(ds_woody_veg, outdir, stub, min_patch_size=min_patch_size, min_core_size=min_core_size, edge_size=edge_size, max_gap_size=max_gap_size, strict_core_area=strict_core_area, save_tif=debug, plot=debug)
    ds_shelter = shelter_categories(ds_tree_categories, wind_data=ds_wind, wind_method=wind_method, wind_threshold=wind_threshold, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=debug, plot=debug, crop_pixels=crop_pixels)
    ds_cover = cover_categories(ds_shelter, da_worldcover, outdir=outdir, stub=stub, savetif=debug, plot=debug)

    ds_buffer = buffer_categories(ds_cover, ds_hydrolines, roads_data=ds_roads, outdir=outdir, stub=stub, buffer_width=buffer_width, savetif=debug, plot=debug)
    ds_linear, df_patches = patch_metrics(ds_buffer, outdir, stub, plot=debug, save_csv=debug, save_labels=False, crop_pixels=crop_pixels, min_shelterbelt_length=min_shelterbelt_length, max_shelterbelt_width=max_shelterbelt_width, min_patch_size=min_patch_size) 

    # Trying to avoid memory accumulation
    for ds in [ds_worldcover, ds_roads, ds_hydrolines, ds_woody_veg, ds_tree_categories, ds_shelter, ds_cover, ds_buffer, ds_linear]:
        try:
            ds.close()
            del ds
        except Exception:
            pass
    del df_patches
    locals().clear()
    gc.collect()
    # rasterio.shutil.delete_raster_cache()
    mem_info = process.memory_full_info()
    # print(f"RSS: {mem_info.rss / 1e9:.2f} GB, VMS: {mem_info.vms / 1e9:.2f} GB, Shared: {mem_info.shared / 1e9:.2f} GB")
    # print("Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "MB")
    # print("Number of open files:", len(psutil.Process(os.getpid()).open_files()))
    return None

def indices_csv(csv, outdir=default_outdir,
                     tmpdir=default_tmpdir, stub=None,
                     wind_method=None, wind_threshold=20,
                     cover_threshold=1, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                     crop_pixels=0, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6):
    """
    Run the indices pipeline for every file listed in a CSV.

    The CSV is expected to contain a column named ``filename`` with full paths
    to percent-cover GeoTIFFs. Each row is processed sequentially by
    `indices_tif` using the provided parameters.

    Parameters
    ----------
    csv : str
        Path to a CSV file containing a `filename` column with input TIFF paths.
    Other parameters
        Passed through to :func:`indices_tif` (see that function for details).

    """
    df = pd.read_csv(csv)
    for percent_tif in df['filename']:
        # The provided stub needs to be None, because we want to use the percent_tif filename instead. 
        indices_tif(percent_tif, outdir, tmpdir, None, wind_method, wind_threshold, cover_threshold, min_patch_size, edge_size, max_gap_size, distance_threshold, density_threshold, buffer_width, strict_core_area, crop_pixels, min_core_size, min_shelterbelt_length, max_shelterbelt_width)


def indices_tifs(folder, outdir=default_outdir, tmpdir=default_tmpdir, param_stub='', 
                      wind_method=None, wind_threshold=20,
                      cover_threshold=1, min_patch_size=20, edge_size=3, max_gap_size=1,
                      distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                      crop_pixels=0, limit=None, tiles_per_csv=100, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6, suffix='tif'):
    """
    Run the indices pipeline over a folder of binary or integer tifs representing percentage tree cover.

    Parameters
    ----------
    folder : str
        Input directory containing binary or integer TIFFs.
    outdir : str, optional
        Output directory for generated linear/category TIFFs.
    tmpdir : str, optional
        Directory used for temporary CSVs and intermediate files.
    param_stub : str, optional
        Extra stub for csv filenames and downstream tifs.
    tiles_per_csv : int, optional
        Number of tiles grouped per subprocess CSV. Default is 100.
    Other parameters
        Passed through to :func:`indices_tif` (see that function for details).

    """
    os.makedirs(outdir, exist_ok=True)
    percent_tifs = glob.glob(f'{folder}/*.{suffix}')
    print(f"Starting with {len(percent_tifs)} percent_tifs", flush=True)

    if limit:
        percent_tifs = percent_tifs[:limit]

    if limit is None: # Don't remove tifs if we've specified a limit, because it's just for testing so I want reproducible results.
        # Remove tifs that have already been processed (sometimes I have to run this multiple times if a process runs out of memory or rasterio gives a parallelisation conflict)
        percent_stubs = [pathlib.Path(tif).stem[:12] for tif in percent_tifs]
        processed = glob.glob(f'{outdir}/*.tif')
        processed_stubs = set(pathlib.Path(tif).stem[:12] for tif in processed)
        percent_tifs = [tif for tif, stub in zip(percent_tifs, percent_stubs) if stub not in processed_stubs]
        print(f"Reduced to {len(percent_tifs)} percent_tifs", flush=True)

    df = pd.DataFrame(percent_tifs, columns=["filename"])
    csv_filenames = []
    chunk_size = tiles_per_csv
    for i in range(math.ceil(len(df) / chunk_size)):
        chunk = df[i*chunk_size : (i+1)*chunk_size]
        all_the_params = f'{wind_method}_w{wind_threshold}_c{cover_threshold}_m{min_patch_size}_e{edge_size}_g{max_gap_size}_di{distance_threshold}_de{density_threshold}_b{buffer_width}_mc{min_core_size}_msl{min_shelterbelt_length}_msw{max_shelterbelt_width}_sca{strict_core_area}' # Anything that might be run in parallel needs a unique filename
        filename = os.path.join(tmpdir, f"{param_stub}_{all_the_params}_run_pipeline_tifs_{i}.csv")
        chunk.to_csv(filename, index=False)
        csv_filenames.append(filename)
        print("Saved:", filename, flush=True)

    for i, filename in enumerate(csv_filenames):
        print(f"Launching Popen subprocess for filename {i}/{len(csv_filenames)}:", filename, flush=True)

        script = os.path.join(os.path.dirname(__file__), "indices.py") # Use the module filename for robustness
        cmd = [
            sys.executable,
            "indices.py", 
            str(filename),
            "--outdir", str(outdir),
            "--tmpdir", str(tmpdir),
            "--param_stub", str(param_stub),  # or args.param_stub if applicable
            "--wind_method", str(wind_method),
            "--wind_threshold", str(wind_threshold),
            "--cover_threshold", str(cover_threshold),
            "--min_patch_size", str(min_patch_size),
            "--edge_size", str(edge_size),
            "--max_gap_size", str(max_gap_size),
            "--distance_threshold", str(distance_threshold),
            "--density_threshold", str(density_threshold),
            "--buffer_width", str(buffer_width),
            "--crop_pixels", str(crop_pixels),
            "--min_core_size", str(min_core_size),
            "--min_shelterbelt_length", str(min_shelterbelt_length),
            "--max_shelterbelt_width", str(max_shelterbelt_width)
        ]
        if not strict_core_area:
            cmd += ["--no-strict-core-area"]
    
        # Popen a subprocess to hopefully avoid memory accumulation
        p = subprocess.Popen(cmd)
        p.wait()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the shelterbelt delineation pipeline on a folder of percent_cover.tifs.")

    parser.add_argument("folder", help="Input folder containing percent_cover.tifs")
    parser.add_argument("--outdir", default=default_outdir, help=f"Output folder for linear_categories.tifs (default: {default_outdir})")
    parser.add_argument("--tmpdir", default=default_tmpdir, help=f"Temporary working folder (default: {default_tmpdir})")
    parser.add_argument("--param_stub", default=None, help="Extra stub for the suffix of the merged tif")
    parser.add_argument("--wind_method", default=None, help="Method used to infer shelter direction")
    parser.add_argument("--wind_threshold", type=int, default=20, help="Wind speed threshold in km/h")
    parser.add_argument("--cover_threshold", type=int, default=1, help="Percentage tree cover within a pixel to classify as tree (default: 1)")
    parser.add_argument("--min_patch_size", type=int, default=20, help="Minimum area (pixels) to classify as a patch rather than scattered trees")
    parser.add_argument("--edge_size", type=int, default=3, help="Distance (pixels) defining the edge region around patch cores")
    parser.add_argument("--max_gap_size", type=int, default=1, help="Maximum gap (pixels) to bridge when connecting tree clusters")
    parser.add_argument("--distance_threshold", type=int, default=20, help="Distance from trees that counts as sheltered")
    parser.add_argument("--density_threshold", type=int, default=5, help="Percentage tree cover within distance_threshold that counts as sheltered")
    parser.add_argument("--buffer_width", type=int, default=3, help="Number of pixels away from the feature that still counts as within the buffer")
    parser.add_argument("--crop_pixels", type=int, default=0, help="Number of pixels to crop from each edge of the output")
    parser.add_argument('--no-strict-core-area', dest='strict_core_area', action='store_false', default=True, help='Disable strict core area enforcement (default: enabled)')
    parser.add_argument("--limit", type=int, default=None, help="Number of tifs to process (default: all)")
    parser.add_argument("--min_core_size", type=int, default=1000, help="Minimum area (pixels) to classify as a core area")
    parser.add_argument("--min_shelterbelt_length", type=int, default=15, help="Minimum skeleton length (in pixels) to classify a cluster as linear")
    parser.add_argument("--max_shelterbelt_width", type=int, default=6, help="Maximum skeleton width (in pixels) to classify a cluster as linear")
    parser.add_argument("--suffix", default='tif', help="Suffix of each of the input tif files")

    return parser


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    if args.folder.endswith('.tif'):
        indices_tif(
            args.folder,
            outdir=args.outdir,
            tmpdir=args.tmpdir,
            stub=args.param_stub,
            wind_method=args.wind_method,
            wind_threshold=args.wind_threshold,
            cover_threshold=args.cover_threshold,
            min_patch_size=args.min_patch_size,
            edge_size=args.edge_size,
            max_gap_size=args.max_gap_size,
            distance_threshold=args.distance_threshold,
            density_threshold=args.density_threshold,
            buffer_width=args.buffer_width,
            strict_core_area=args.strict_core_area,
            crop_pixels=args.crop_pixels,
            min_core_size=args.min_core_size,
            min_shelterbelt_length=args.min_shelterbelt_length,
            max_shelterbelt_width=args.max_shelterbelt_width
        )
    elif args.folder.endswith('.csv'):
            indices_csv(
            args.folder,
            outdir=args.outdir,
            tmpdir=args.tmpdir,
            stub=args.param_stub,
            wind_method=args.wind_method,
            wind_threshold=args.wind_threshold,
            cover_threshold=args.cover_threshold,
            min_patch_size=args.min_patch_size,
            edge_size=args.edge_size,
            max_gap_size=args.max_gap_size,
            distance_threshold=args.distance_threshold,
            density_threshold=args.density_threshold,
            buffer_width=args.buffer_width,
            strict_core_area=args.strict_core_area,
            crop_pixels=args.crop_pixels,
            min_core_size=args.min_core_size,
            min_shelterbelt_length=args.min_shelterbelt_length,
            max_shelterbelt_width=args.max_shelterbelt_width
        )
    else:
        indices_tifs(
            folder=args.folder,
            outdir=args.outdir,
            tmpdir=args.tmpdir,
            param_stub=args.param_stub,
            wind_method=args.wind_method,
            wind_threshold=args.wind_threshold,
            cover_threshold=args.cover_threshold,
            min_patch_size=args.min_patch_size,
            edge_size=args.edge_size,
            max_gap_size=args.max_gap_size,
            distance_threshold=args.distance_threshold,
            density_threshold=args.density_threshold,
            buffer_width=args.buffer_width,
            strict_core_area=args.strict_core_area,
            crop_pixels=args.crop_pixels,
            limit=args.limit,
            min_core_size=args.min_core_size,
            min_shelterbelt_length=args.min_shelterbelt_length,
            max_shelterbelt_width=args.max_shelterbelt_width,
            suffix=args.suffix
        )
