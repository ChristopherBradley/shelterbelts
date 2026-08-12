import os
import glob
import argparse
import math
import pathlib

import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import box

# Trying to avoid memory issues
import gc
import subprocess, sys

from shelterbelts.utils.tiles import merge_tiles_bbox, merged_ds, crop_and_rasterize
from shelterbelts.apis.barra_daily import barra_daily
from shelterbelts.apis.canopy_height import canopy_height
from shelterbelts.apis.worldcover import worldcover_centrepoint
from shelterbelts.apis.osm import osm_roads

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.patch_metrics import patch_metrics
from shelterbelts.indices.shelter_categories import shelter_categories, shelter_categories_cmap, shelter_categories_labels
from shelterbelts.indices.catchments import catchments
from shelterbelts.indices.opportunities import opportunities_da

# 11 secs for all these imports
# -
from shelterbelts.utils.filepaths import (
    worldcover_dir,
    worldcover_geojson,
    hydrolines_gdb,
    roads_gdb,
    IS_GADI,
    ensure_env_on_path,
)
from shelterbelts.utils.visualisation import tif_categorical, visualise_categories


def opportunity_shelter(ds_opportunities, ds_linear, ds_shelter, ds_wind=None, wind_method=None,
                        wind_threshold=20, distance_threshold=20, density_threshold=5,
                        outdir='.', stub='TEST', savetif=True, plot=False):
    """Calculate farmland that would become sheltered if trees were planted at the opportunity locations.

    Parameters
    ----------
    ds_opportunities : xarray.Dataset
        Output of :func:`opportunities_da`
    ds_linear : xarray.Dataset
        The 'linear_categories' classification the original shelter was derived from.
    ds_shelter : xarray.Dataset
        The original 'shelter_categories'.
    ds_wind, wind_method, wind_threshold, distance_threshold, density_threshold
        Passed to :func:`shelter_categories`; use the same values as the original shelter step.
    outdir, stub, savetif, plot
        Where/whether to save the combined ``{stub}_opportunities.tif`` and PNG.

    Returns
    -------
    xarray.Dataset
        ``ds_opportunities`` with the would-be-sheltered farmland codes (3X grassland, 4X cropland,
        where X is the sheltering opportunity tree) merged into 'opportunities'.
    """
    da_opp = ds_opportunities['opportunities']
    opportunity_trees = da_opp > 0

    if bool(opportunity_trees.any()):
        # Add the opportunity pixels as trees on a copy of linear trees before re-running the shelter-categories
        da_linear = ds_linear['linear_categories']
        da_planted = xr.where(opportunity_trees, da_opp, da_linear).astype('uint8').rio.write_crs(da_linear.rio.crs)

        ds_planted = shelter_categories(
            da_planted.to_dataset(name='linear_categories'), wind_data=ds_wind, wind_method=wind_method,
            wind_threshold=wind_threshold, distance_threshold=distance_threshold,
            density_threshold=density_threshold, savetif=False, plot=False)
        new = ds_planted['shelter_categories']
        orig = ds_shelter['shelter_categories']

        # Sheltered grassland is encoded 32-39 and sheltered cropland 42-49 (see shelter_categories_labels).
        newly_grassland = (new >= 32) & (new <= 39) & ~((orig >= 32) & (orig <= 39)) & (da_opp == 0)
        newly_cropland = (new >= 42) & (new <= 49) & ~((orig >= 42) & (orig <= 49)) & (da_opp == 0)

        da_opp = xr.where(newly_grassland, new, da_opp)
        da_opp = xr.where(newly_cropland, new, da_opp)
        ds_opportunities['opportunities'] = da_opp.astype('uint8').rio.write_crs(ds_opportunities['opportunities'].rio.crs)

    if savetif:
        filename = os.path.join(outdir, f"{stub}_opportunities.tif")
        tif_categorical(ds_opportunities['opportunities'], filename, shelter_categories_cmap)
    if plot:
        filename_png = os.path.join(outdir, f"{stub}_opportunities.png")
        visualise_categories(ds_opportunities['opportunities'], filename_png, shelter_categories_cmap, shelter_categories_labels, "Opportunities")

    return ds_opportunities


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
  30: 'Unsheltered Grassland',
  31: 'Unsheltered Grassland',
  32: 'Sheltered Grassland',
  33: 'Grassland sheltered by Patch Edge',
  34: 'Grassland sheltered by Other Trees',
  35: 'Grassland sheltered by Trees in Gullies',
  36: 'Grassland sheltered by Trees on Ridges',
  37: 'Grassland sheltered by Trees next to Roads',
  38: 'Grassland sheltered by Linear Patches',
  39: 'Grassland sheltered by Non-linear Patches',
  40: 'Unsheltered Cropland',
  41: 'Unsheltered Cropland',
  42: 'Sheltered Cropland',
  43: 'Cropland sheltered by Patch Edge',
  44: 'Cropland sheltered by Other Trees',
  45: 'Cropland sheltered by Trees in Gullies',
  46: 'Cropland sheltered by Trees on Ridges',
  47: 'Cropland sheltered by Trees next to Roads',
  48: 'Cropland sheltered by Linear Patches',
  49: 'Cropland sheltered by Non-linear Patches',
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
                     opportunities=False, height_tif=None, save_gpkg=False, save_patch_csv=False,
                     worldcover_data=None, gullies_data=None, roads_data=None,
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
    opportunities : bool, optional
        Generate an opportunities tif showing tree and shelter opportunities near gullies and roads.
        Uses buffer_width as the buffer around roads/gullies for planting opportunities.
    height_tif : str, optional
        Path to a canopy-height GeoTIFF (metres). When provided, the distance_threshold uses a relative distance in tree heights,
        rather than an absolute distance in pixels.
    save_gpkg : bool, optional
        Save the per-tile shelterbelt centrelines vector ``{stub}_centrelines.gpkg`` from patch_metrics.
    save_patch_csv : bool, optional
        Save the per-tile ``{stub}_patch_metrics.csv``.
    worldcover_data, gullies_data, roads_data : optional
        Pre-loaded WorldCover DataArray / gullies Dataset / roads Dataset to use instead of loading
        them from the Australian tile and GDB sources. Lets callers such as :func:`indices_latlon`
        reuse this pipeline with their own data. When None (the default) they are loaded here.
    debug : bool, optional
        If True, intermediate TIFFs/plots are saved for debugging.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with shelter_categories, linear_categories and labelled_categories bands.
    df : pandas.DataFrame
        Per-cluster patch metrics (skeleton length/width, category, etc.).

    Examples
    --------
    .. plot::

        import matplotlib.pyplot as plt
        import rioxarray as rxr
        from shelterbelts.indices.all_indices import indices_tif
        from shelterbelts.indices.shelter_categories import shelter_categories_cmap, shelter_categories_labels
        from shelterbelts.utils.visualisation import _plot_categories_on_axis
        from shelterbelts.utils.filepaths import get_filename

        tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
        da_trees = rxr.open_rasterio(tree_file).squeeze('band').drop_vars('band')
        tree_cmap = {0: (255, 255, 255), 1: (14, 138, 0)}
        tree_labels = {0: 'No Trees', 1: 'Woody Vegetation'}

        ds_shelter, _ = indices_tif(tree_file, outdir='/tmp', stub='test')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))
        _plot_categories_on_axis(ax1, da_trees, tree_cmap, tree_labels, 'Example Input', legend_inside=True)
        _plot_categories_on_axis(ax2, ds_shelter['shelter_categories'], shelter_categories_cmap, shelter_categories_labels, 'Example Output', legend_inside=True)
        plt.tight_layout()

    """
    if stub is None:
        # stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'
        stub = os.path.basename(percent_tif).split('.')[0][:50] # Hopefully there's something unique in the first 50 characters
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
    
    # Anything that might be run in parallel needs a unique filename, so we don't get rasterio merge conflicts
    worldcover_stub = f'{data_folder}_{stub}_{wind_method}_w{wind_threshold}_c{cover_threshold}_m{min_patch_size}_e{edge_size}_g{max_gap_size}_di{distance_threshold}_de{density_threshold}_b{buffer_width}_mc{min_core_size}_msl{min_shelterbelt_length}_msw{max_shelterbelt_width}_sca{strict_core_area}' # 
    
    if worldcover_data is not None:
        da_worldcover = worldcover_data
        ds_worldcover = None
    else:
        mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, worldcover_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False)
        ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
        da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})

    if gullies_data is not None:
        ds_hydrolines = gullies_data
    else:
        gdf_hydrolines, ds_hydrolines = crop_and_rasterize(da_percent, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, feature_name='gullies')

    if roads_data is not None:
        ds_roads = roads_data
    else:
        gdf_roads, ds_roads = crop_and_rasterize(da_percent, roads_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, layer='NationalRoads_2025_09', feature_name='roads')

    if wind_method and wind_method not in ("None", "MULTI_LAYER"):  # Handling conversion of None to "None" when using subprocess. MULTI_LAYER computes all 8 directions itself and needs no wind data.
        lat = (bbox_4326[1] + bbox_4326[3])/2
        lon = (bbox_4326[0] + bbox_4326[2])/2
        ds_wind = barra_daily(lat=lat, lon=lon, start_year=2020, end_year=2020, gdata=IS_GADI, plot=False, save_netcdf=False) # This line is currently the limiting factor since it takes 4 secs
    else:
        # if no wind_method provided than the percent_cover method without wind gets used
        ds_wind = None

    ds_woody_veg = da_trees.to_dataset(name='woody_veg')
    ds_tree_categories = tree_categories(ds_woody_veg, outdir, stub, min_patch_size=min_patch_size, min_core_size=min_core_size, edge_size=edge_size, max_gap_size=max_gap_size, strict_core_area=strict_core_area, save_tif=debug, plot=debug)
    ds_cover = cover_categories(ds_tree_categories, da_worldcover, outdir=outdir, stub=stub, savetif=debug, plot=debug)
    ds_buffer = buffer_categories(ds_cover, ds_hydrolines, roads_data=ds_roads, outdir=outdir, stub=stub, buffer_width=buffer_width, savetif=debug, plot=debug)
    ds_linear, df_patches = patch_metrics(ds_buffer, outdir, stub, plot=debug, save_csv=(save_patch_csv or debug), save_labels=False, save_gpkg=save_gpkg, crop_pixels=crop_pixels, min_shelterbelt_length=min_shelterbelt_length, max_shelterbelt_width=max_shelterbelt_width, min_patch_size=min_patch_size, max_gap_size=max_gap_size)
    ds_shelter = shelter_categories(ds_linear, wind_data=ds_wind, height_tif=height_tif, wind_method=wind_method, wind_threshold=wind_threshold, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=True, plot=debug)

    if opportunities:
        # Reproject the layers onto the tree grid
        da_worldcover_matched = da_worldcover.rio.reproject_match(da_trees)

        # Casting to uint8 because rioxarray.reproject_match can't derive a nodata value for bool masks
        da_roads_opp = ds_roads['roads'].astype('uint8').rio.reproject_match(da_trees)
        da_gullies_opp = ds_hydrolines['gullies'].astype('uint8').rio.reproject_match(da_trees)
        ds_opportunities = opportunities_da(
            da_trees.astype('uint8'), da_roads_opp, da_gullies_opp,
            None, None, da_worldcover_matched,          # da_ridges=None, da_dem=None, since these don't seem as reliable/interesting yet.
            outdir=outdir, stub=stub, tmpdir=tmpdir,
            width=buffer_width, contour_spacing=0,
            savetif=False, plot=False, crop_pixels=crop_pixels,
        )
        opportunity_shelter(
            ds_opportunities, ds_linear, ds_shelter, ds_wind=ds_wind, wind_method=wind_method,
            wind_threshold=wind_threshold, distance_threshold=distance_threshold, density_threshold=density_threshold,
            outdir=outdir, stub=stub, savetif=True, plot=debug,
        )

    # Trying to avoid memory accumulation
    for ds in [ds_worldcover, ds_roads, ds_hydrolines, ds_woody_veg, ds_tree_categories, ds_cover, ds_buffer, ds_linear]:
        try:
            ds.close()
            del ds
        except Exception:
            pass
    gc.collect()
    return ds_shelter, df_patches

_AUSTRALIA_BOUNDS = (-44, 113, -10, 154)  # (lat_min, lon_min, lat_max, lon_max)


def indices_latlon(lat, lon, buffer=0.05, outdir=".", tmpdir=".", stub=None,
                   wind_method=None, wind_threshold=20,
                   height_threshold=1.0, cover_threshold=1,
                   min_patch_size=20, edge_size=3, max_gap_size=1,
                   distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                   crop_pixels=0, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6,
                   opportunities=False,
                   debug=False):
    """
    Run the complete indices pipeline for a lat/lon location, auto-downloading all required data.

    Downloads canopy height (Meta/Tolan global CHM), ESA WorldCover, gullies (Australian hydrolines
    GDB or, outside Australia, terrain tiles), roads (Australian NationalRoads GDB or OpenStreetMap),
    and BARRA if the wind_method is set.

    Parameters
    ----------
    lat : float
        Latitude in WGS 84 (EPSG:4326).
    lon : float
        Longitude in WGS 84 (EPSG:4326).
    buffer : float, optional
        Half-width of the region of interest in degrees (~5 km at 0.05).
    stub : str, optional
        Prefix for output filenames.
    height_threshold : float, optional
        Canopy height (metres) above which a 1 m pixel is classified as tree.
    cover_threshold : int, optional
        Minimum percentage of tree-pixels within a 10 m cell to count it as tree.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with shelter_categories, linear_categories and labelled_categories bands.
    df : pandas.DataFrame
        Per-cluster patch metrics.

    Notes
    -----
    In Australia the higher-quality Geoscience Australia GDB vectors are used automatically when they
    exist and cover the point: gullies from shelterbelts.utils.filepaths.hydrolines_gdb and roads from
    roads_gdb. Otherwise (outside Australia, the GDB not installed, or the GDB has no features for this
    point) gullies are derived from downloaded terrain tiles and roads from OpenStreetMap.
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

    # 3. Average-resample the 1 m canopy height model to a 10m percent cover tif
    da_trees_pct = da_trees_1m.rio.reproject_match(da_worldcover, resampling=Resampling.average) * 100
    percent_tif = os.path.join(tmpdir, f"{stub}_trees_percent.tif")
    da_trees_pct.rio.to_raster(percent_tif)
    da_trees = da_trees_pct >= cover_threshold   # only used for rasterizing the OSM roads onto the tree grid

    lat_min, lon_min, lat_max, lon_max = _AUSTRALIA_BOUNDS
    in_australia = lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    # 4. Gullies: choose the National hydrolines GDB when the point is inside Australia,
    #    otherwise derive them from downloaded terrain tiles.
    ds_gullies = None
    if in_australia and os.path.exists(hydrolines_gdb):
        _, ds_gul = crop_and_rasterize(da_trees, hydrolines_gdb, outdir=tmpdir, stub=stub,
            savetif=debug, save_gpkg=debug, feature_name='gullies')
        if bool(ds_gul['gullies'].values.any()):
            ds_gullies = ds_gul
        else:
            print("No hydrolines cover this point in hydrolines_gdb - generating from DEM instead. "
                  "Tip: For more accurate gullies in Australia, download the National Surface Hydrolines GDB from Geoscience and specify the path in utils.filepaths.hydrolines_gdb.")
    if ds_gullies is None:
        # terrain_tiles calls gdalwarp as a subprocess so we need to ensure the conda env bin and PROJ db are findable
        ensure_env_on_path()
        terrain_tiles(lat, lon, buffer, outdir=tmpdir, stub=stub, tmpdir=tmpdir, verbose=debug)
        terrain_tif = os.path.join(tmpdir, f"{stub}_terrain.tif")
        ds_catch = catchments(terrain_tif, outdir=tmpdir, stub=stub, savetif=debug, plot=debug)
        # The catchment gullies are a bool mask, so cast to uint8 so reproject_match can derive a nodata value.
        ds_gullies = ds_catch['gullies'].astype('uint8').to_dataset(name='gullies')

    # 5. Roads: choose NationalRoads GDB when the point is in Australia,
    #    otherwise fall back to OpenStreetMap.
    ds_roads = None
    if in_australia and os.path.exists(roads_gdb):
        _, ds_rd = crop_and_rasterize(da_trees, roads_gdb, outdir=tmpdir, stub=stub,
            savetif=debug, save_gpkg=debug, layer='NationalRoads_2025_09', feature_name='roads')
        if bool(ds_rd['roads'].values.any()):
            ds_roads = ds_rd
        else:
            print("No roads cover this point in roads_gdb — falling back to OpenStreetMap. "
                  "Tip: For better compute efficiency in Australia, you can pre-download the full NationalRoads GDB and set the path in utils.filepaths.roads_gdb.")
    if ds_roads is None:
        _, ds_roads = osm_roads(da_trees, outdir=tmpdir, stub=stub, savetif=debug, save_gpkg=debug)

    # 6. Run the indices_tif pipeline
    return indices_tif(
        percent_tif, outdir=outdir, tmpdir=tmpdir, stub=stub,
        wind_method=wind_method, wind_threshold=wind_threshold, cover_threshold=cover_threshold,
        min_patch_size=min_patch_size, edge_size=edge_size, max_gap_size=max_gap_size,
        distance_threshold=distance_threshold, density_threshold=density_threshold,
        buffer_width=buffer_width, strict_core_area=strict_core_area, crop_pixels=crop_pixels,
        min_core_size=min_core_size, min_shelterbelt_length=min_shelterbelt_length,
        max_shelterbelt_width=max_shelterbelt_width, opportunities=opportunities,
        worldcover_data=da_worldcover, gullies_data=ds_gullies, roads_data=ds_roads, debug=debug,
    )


def indices_csv(csv, outdir=".",
                     tmpdir=".", stub=None,
                     wind_method=None, wind_threshold=20,
                     cover_threshold=1, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                     crop_pixels=0, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6,
                     opportunities=False, height_tif=None, save_gpkg=False, save_patch_csv=False,
                     debug=False):
    """
    Run the indices pipeline for every file listed in a CSV.

    The CSV is expected to contain a column named filename with full paths
    to percent-cover GeoTIFFs. Each row is processed sequentially by
    `indices_tif` using the provided parameters.

    Parameters
    ----------
    csv : str
        Path to a CSV file containing a `filename` column with input TIFF paths.
        An optional `height_tif` column may give a per-tile canopy-height GeoTIFF; when present
        it overrides the height_tif argument for that row.
    Other parameters
        Passed through to :func:`indices_tif` (see that function for details).

    """
    df = pd.read_csv(csv)
    has_height_col = 'height_tif' in df.columns
    n_ok, n_fail = 0, 0
    for row in df.itertuples(index=False):
        percent_tif = row.filename
        # Per-row height tif from the CSV takes precedence, otherwise fall back to the shared argument.
        row_height_tif = getattr(row, 'height_tif', None) if has_height_col else height_tif
        if isinstance(row_height_tif, float) and pd.isna(row_height_tif):
            row_height_tif = None
        # Isolate each tile so one degenerate tile doesn't abort the rest of the region's tiles.
        try:
            # The provided stub is None, so we can use the percent_tif filename instead.
            indices_tif(percent_tif, outdir, tmpdir, None, wind_method, wind_threshold, cover_threshold, min_patch_size, edge_size, max_gap_size, distance_threshold, density_threshold, buffer_width, strict_core_area, crop_pixels, min_core_size, min_shelterbelt_length, max_shelterbelt_width, opportunities=opportunities, height_tif=row_height_tif, save_gpkg=save_gpkg, save_patch_csv=save_patch_csv, debug=debug)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"SKIPPED tile (error): {percent_tif}: {type(exc).__name__}: {exc}", flush=True)
    print(f"indices_csv finished: {n_ok} ok, {n_fail} skipped", flush=True)


def indices_tifs(folder, outdir=".", tmpdir=".", param_stub='',
                      wind_method=None, wind_threshold=20,
                      cover_threshold=1, min_patch_size=20, edge_size=3, max_gap_size=1,
                      distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                      crop_pixels=0, limit=None, tiles_per_csv=1000, min_core_size=1000, min_shelterbelt_length=15, max_shelterbelt_width=6, suffix='tif',
                      opportunities=False, height_dir=None, save_gpkg=False, save_patch_csv=False,
                      debug=False):
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
    height_dir : str, optional
        Directory of per-tile canopy-height tifs, named identically to the percent-cover tifs in ``folder``. 
    Other parameters
        Passed through to :func:`indices_tif` (see that function for details).

    """
    os.makedirs(outdir, exist_ok=True)
    percent_tifs = glob.glob(f'{folder}/*.{suffix}')
    print(f"Starting with {len(percent_tifs)} percent_tifs", flush=True)

    if limit:
        percent_tifs = percent_tifs[:limit]

    if limit is None: # Don't remove tifs if we've specified a limit because this argument is just used for testing when I want reproducible results.
        # Remove tifs that have already been processed (sometimes I have to run the pipeline multiple times if jobs don't finish)
        processed = glob.glob(f'{outdir}/*.tif')
        processed_stems = [pathlib.Path(tif).stem for tif in processed]
        percent_tifs = [
            tif for tif in percent_tifs
            if not any(s.startswith(pathlib.Path(tif).stem[:50]) for s in processed_stems)
        ]
        print(f"Reduced to {len(percent_tifs)} percent_tifs", flush=True)

    df = pd.DataFrame(percent_tifs, columns=["filename"])
    if height_dir is not None:
        # Match each percent tif to a same-named height tif so indices_csv can pass it per-row.
        df["height_tif"] = [os.path.join(height_dir, os.path.basename(t)) for t in df["filename"]]
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
            "--max_shelterbelt_width", str(max_shelterbelt_width),
        ]
        if not strict_core_area:
            cmd += ["--no-strict-core-area"]
        if opportunities:
            cmd += ["--opportunities"]
        if save_gpkg:
            cmd += ["--save_gpkg"]
        if save_patch_csv:
            cmd += ["--save_patch_csv"]
        if debug:
            cmd += ["--debug"]
    
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
    parser.add_argument("--height_tif", default=None, help="Canopy-height GeoTIFF (metres) for a single input tif; shelter distance is then measured in tree heights")
    parser.add_argument("--height_dir", default=None, help="Directory of per-tile canopy-height GeoTIFFs (named identically to the input tifs) for folder mode")
    parser.add_argument('--opportunities', action='store_true', default=False, help='Also classify tree-planting opportunities near roads and gullies (default: False)')
    parser.add_argument('--save_gpkg', action='store_true', default=False, help='Save per-tile shelterbelt centrelines GeoPackage from patch_metrics (default: False)')
    parser.add_argument('--save_patch_csv', action='store_true', default=False, help='Save per-tile patch_metrics.csv so combine_patch_metrics_csvs can stamp the source tile (default: False)')
    parser.add_argument('--debug', action='store_true', default=False, help='Save intermediate TIFFs and plots for debugging (default: False)')

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
            max_shelterbelt_width=args.max_shelterbelt_width,
            opportunities=args.opportunities,
            height_tif=args.height_tif,
            save_gpkg=args.save_gpkg,
            save_patch_csv=args.save_patch_csv,
            debug=args.debug,
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
            max_shelterbelt_width=args.max_shelterbelt_width,
            opportunities=args.opportunities,
            height_tif=args.height_tif,
            save_gpkg=args.save_gpkg,
            save_patch_csv=args.save_patch_csv,
            debug=args.debug,
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
            suffix=args.suffix,
            opportunities=args.opportunities,
            height_dir=args.height_dir,
            save_gpkg=args.save_gpkg,
            save_patch_csv=args.save_patch_csv,
            debug=args.debug,
        )
