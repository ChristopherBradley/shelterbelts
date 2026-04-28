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
from shelterbelts.apis.canopy_height import canopy_height
from shelterbelts.apis.worldcover import worldcover_centrepoint
from shelterbelts.apis.osm import osm_roads

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import patch_metrics
from shelterbelts.indices.catchments import catchments, gullies_cmap, ridges_cmap

# 11 secs for all these imports
# -
from shelterbelts.utils.filepaths import (
    worldcover_dir,
    worldcover_geojson,
    hydrolines_gdb,
    roads_gdb,
    IS_GADI,
)
from shelterbelts.utils.visualisation import tif_categorical

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

def indices_tif(percent_tif, outdir=".",
                     tmpdir=".", stub=None,
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
        Prefix for output filenames. If not provided it is derived from percent_tif.
    wind_method : str or None, optional
        Method used to infer shelter direction. Can be None,
        'WINDWARD', 'MOST_COMMON', 'MAX', 'HAPPENED' or
        'ANY'. See :func:`shelter_categories` for details.
    wind_threshold : int, optional
        Wind speed threshold in km/h.
    cover_threshold : int, optional
        Pixel percent cover threshold to treat a pixel as 'tree'.
        - If input is a binary tif use cover_threshold=1.
        - For percent-cover tifs typical values are 10 or 20.
        - For confidence tifs a value like 50 is reasonable.
    min_patch_size : int, optional
        Minimum area (pixels) to classify as a patch rather than scattered trees.
    edge_size : int, optional
        Distance (pixels) defining the edge region around patch cores.
    max_gap_size : int, optional
        Maximum gap (pixels) to bridge when connecting tree clusters.
    distance_threshold : int, optional
        Distance from trees that counts as sheltered.
        Units are either 'tree heights' or 'number of pixels', depending on if a height_tif is provided.
    density_threshold : int, optional
        Percentage tree cover within the distance_threshold that counts as sheltered.
        Only applies if the wind_data is not provided.
    buffer_width : int, optional
        Number of pixels away from the feature that still counts as within the buffer.
    strict_core_area : bool, optional
        If True, enforce that core areas exceed the edge_size at all points.
        If False, use dilation and erosion to allow some irregularity.
    crop_pixels : int, optional
        Number of pixels to crop from each edge of the output.
    min_core_size : int, optional
        Minimum area (pixels) to classify as a core area.
    min_shelterbelt_length : int, optional
        Minimum skeleton length (in pixels) to classify a cluster as linear.
    max_shelterbelt_width : int, optional
        Maximum skeleton width (in pixels) to classify a cluster as linear.
    debug : bool, optional
        If True, intermediate TIFFs/plots are saved for debugging.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with linear_categories and labelled_categories bands.
    df : pandas.DataFrame
        Per-cluster patch metrics (skeleton length/width, category, etc.).

    Examples
    --------
    .. plot::

        import matplotlib.pyplot as plt
        import rioxarray as rxr
        from shelterbelts.indices.all_indices import indices_tif
        from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels
        from shelterbelts.utils.visualisation import _plot_categories_on_axis
        from shelterbelts.utils.filepaths import get_filename

        tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
        da_trees = rxr.open_rasterio(tree_file).squeeze('band').drop_vars('band')
        tree_cmap = {0: (255, 255, 255), 1: (14, 138, 0)}
        tree_labels = {0: 'No Trees', 1: 'Woody Vegetation'}

        ds_linear, _ = indices_tif(tree_file, outdir='/tmp', stub='test')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))
        _plot_categories_on_axis(ax1, da_trees, tree_cmap, tree_labels, 'Example Input', legend_inside=True)
        _plot_categories_on_axis(ax2, ds_linear['linear_categories'], linear_categories_cmap, linear_categories_labels, 'Example Output', legend_inside=True)
        plt.tight_layout()

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
    for ds in [ds_worldcover, ds_roads, ds_hydrolines, ds_woody_veg, ds_tree_categories, ds_shelter, ds_cover, ds_buffer]:
        try:
            ds.close()
            del ds
        except Exception:
            pass
    gc.collect()
    mem_info = process.memory_full_info()
    return ds_linear, df_patches

_AUSTRALIA_BOUNDS = (-44, 113, -10, 154)  # (lat_min, lon_min, lat_max, lon_max)


def indices_latlon(lat, lon, buffer=0.05, outdir=".", tmpdir=".", stub=None,
                   wind_method=None, wind_threshold=20,
                   height_threshold=1.0, cover_threshold=1,
                   min_patch_size=20, edge_size=3, max_gap_size=1,
                   distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                   crop_pixels=0, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6,
                   debug=False):
    """
    Run the complete indices pipeline for a lat/lon location, auto-downloading all required data.

    Downloads canopy height (Meta/Tolan global CHM), ESA WorldCover, terrain tiles for
    gully/ridge delineation, and OpenStreetMap roads. BARRA wind data is only downloaded
    when wind_method is set.

    Parameters
    ----------
    lat : float
        Latitude in WGS 84 (EPSG:4326).
    lon : float
        Longitude in WGS 84 (EPSG:4326).
    buffer : float, optional
        Half-width of the region of interest in degrees (~5 km at 0.05).
    outdir : str, optional
        Output directory for saving results.
    tmpdir : str, optional
        Directory for temporary/cached files.
    stub : str, optional
        Prefix for output filenames. Defaults to "{lat:.3f}_{lon:.3f}".
    wind_method : str or None, optional
        Method used to infer shelter direction. See :func:`indices_tif` for options.
    wind_threshold : int, optional
        Wind speed threshold in km/h.
    height_threshold : float, optional
        Canopy height (metres) above which a 1 m pixel is classified as tree.
    cover_threshold : int, optional
        Minimum percentage of tree-pixels within a 10 m cell to count it as tree.
        The 1 m binary raster is average-resampled to 10 m (giving 0–100 % cover) before
        this threshold is applied, matching the behaviour of :func:`indices_tif`.
    min_patch_size, edge_size, max_gap_size, distance_threshold, density_threshold, buffer_width, strict_core_area, crop_pixels, min_core_size, min_shelterbelt_length, max_shelterbelt_width, debug : optional
        Same as :func:`indices_tif`.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with linear_categories and labelled_categories bands.
    df : pandas.DataFrame
        Per-cluster patch metrics.

    Notes
    -----
    For locations in Australia, higher-quality roads and hydrolines are available from the Geoscience
    NationalRoads GDB and SurfaceHydrologyLinesRegional GDB. Set
    shelterbelts.utils.filepaths.roads_gdb and hydrolines_gdb to use them.
    """
    from rasterio.enums import Resampling
    from DAESIM_preprocess.terrain_tiles import terrain_tiles

    if stub is None:
        stub = f"{lat:.3f}_{lon:.3f}"

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    # 1. Canopy height → binary trees at 1 m resolution (EPSG:4326)
    ds_chm = canopy_height(lat, lon, buffer, outdir=tmpdir, stub=stub, save_tif=debug, plot=debug)
    da_trees_1m = (ds_chm['canopy_height'] >= height_threshold).astype(float)

    # 2. WorldCover (EPSG:4326) — provides the reference 10 m grid
    da_worldcover = worldcover_centrepoint(lat, lon, buffer)

    # 3. Average-resample 1 m binary → 0–100 % cover at 10 m, then threshold
    da_trees_pct = da_trees_1m.rio.reproject_match(da_worldcover, resampling=Resampling.average) * 100
    da_trees = da_trees_pct >= cover_threshold
    ds_woody_veg = da_trees.to_dataset(name='woody_veg')

    # 4. DEM → gullies + ridges
    # terrain_tiles calls gdalwarp as a subprocess; ensure it can find the PROJ database
    if 'PROJ_DATA' not in os.environ:
        import sys
        _proj_data = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'proj')
        if os.path.exists(_proj_data):
            os.environ['PROJ_DATA'] = _proj_data
    terrain_tiles(lat, lon, buffer, outdir=tmpdir, stub=stub, tmpdir=tmpdir, verbose=debug)
    terrain_tif = os.path.join(tmpdir, f"{stub}_terrain.tif")
    ds_catch = catchments(terrain_tif, outdir=tmpdir, stub=stub, savetif=debug, plot=debug)
    gullies_tif = os.path.join(tmpdir, f"{stub}_gullies.tif")
    # ridges_tif = os.path.join(tmpdir, f"{stub}_ridges.tif")
    tif_categorical(ds_catch['gullies'], gullies_tif, colormap=gullies_cmap)
    # tif_categorical(ds_catch['ridges'], ridges_tif, colormap=ridges_cmap)

    # 5. Roads via OSM; note better Australian data if relevant
    lat_min, lon_min, lat_max, lon_max = _AUSTRALIA_BOUNDS
    if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
        if not (os.path.exists(hydrolines_gdb) and os.path.exists(roads_gdb)):
            print(
                "Tip: For higher-quality roads and hydrolines in Australia, download the "
                "NationalRoads GDB and SurfaceHydrologyLinesRegional GDB and set "
                "shelterbelts.utils.filepaths.roads_gdb / hydrolines_gdb."
            )
    _, ds_roads = osm_roads(da_trees, outdir=tmpdir, stub=stub, savetif=debug, save_gpkg=debug)

    # 6. Wind — only downloaded when wind_method is set
    if wind_method and wind_method != "None":
        ds_wind = barra_daily(lat=lat, lon=lon, start_year=2020, end_year=2020,
                              gdata=IS_GADI, plot=debug, save_netcdf=debug)
    else:
        ds_wind = None

    # 7. Pipeline (mirrors indices_tif, worldcover already in memory)
    ds_tree_categories = tree_categories(ds_woody_veg, outdir, stub, min_patch_size=min_patch_size,
        min_core_size=min_core_size, edge_size=edge_size, max_gap_size=max_gap_size,
        strict_core_area=strict_core_area, save_tif=debug, plot=debug)
    ds_shelter = shelter_categories(ds_tree_categories, wind_data=ds_wind,
        wind_method=wind_method, wind_threshold=wind_threshold,
        distance_threshold=distance_threshold, density_threshold=density_threshold,
        outdir=outdir, stub=stub, savetif=debug, plot=debug, crop_pixels=crop_pixels)
    ds_cover = cover_categories(ds_shelter, da_worldcover, outdir=outdir, stub=stub,
        savetif=debug, plot=debug)
    ds_buffer = buffer_categories(ds_cover, gullies_tif, ridges_data=None,
        roads_data=ds_roads, outdir=outdir, stub=stub, buffer_width=buffer_width,
        savetif=debug, plot=debug)
    ds_linear, df_patches = patch_metrics(ds_buffer, outdir, stub, plot=debug,
        save_csv=debug, save_labels=debug, crop_pixels=crop_pixels,
        min_shelterbelt_length=min_shelterbelt_length,
        max_shelterbelt_width=max_shelterbelt_width, min_patch_size=min_patch_size)

    return ds_linear, df_patches


def indices_csv(csv, outdir=".",
                     tmpdir=".", stub=None,
                     wind_method=None, wind_threshold=20,
                     cover_threshold=1, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                     crop_pixels=0, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6):
    """
    Run the indices pipeline for every file listed in a CSV.

    The CSV is expected to contain a column named filename with full paths
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


def indices_tifs(folder, outdir=".", tmpdir=".", param_stub='',
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
        Number of tiles grouped per subprocess CSV.
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

        script = os.path.join(os.path.dirname(__file__), "all_indices.py") # Use the module filename for robustness
        cmd = [
            sys.executable,
            script,
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
    parser.add_argument("--outdir", default=".", help="Output folder for linear_categories.tifs (default: current directory)")
    parser.add_argument("--tmpdir", default=".", help="Temporary working folder (default: current directory)")
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
