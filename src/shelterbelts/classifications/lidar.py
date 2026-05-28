# Uses PDAL bindings — requires the shelterbelts conda environment (which bundles python-pdal + rasterio).
import os
import re
import glob
import gc
import json
import argparse
import subprocess

import geopandas as gpd

import numpy as np
import pdal
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box as shapely_box


from shelterbelts.utils.visualisation import tif_categorical
from shelterbelts.classifications.binary_trees import cmap_woody_veg
from shelterbelts.classifications._crown_dalponteCIRC_numba import crowns_to_gpkg
from shelterbelts.classifications.bounding_boxes import bounding_boxes as _bounding_boxes


def _get_laz_bounds(laz_file, epsg=None):
    """Read LAZ header bounds via pdal info."""
    result = subprocess.run(
        ['pdal', 'info', '--metadata', laz_file],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(result.stdout)['metadata']
    minx, miny, maxx, maxy = meta['minx'], meta['miny'], meta['maxx'], meta['maxy']
    if epsg:
        crs = f"EPSG:{epsg}"
    elif 'srs' in meta and meta['srs'].get('wkt'):
        crs = meta['srs']['wkt']
    else:
        crs = meta.get('spatialreference', '')
    return minx, miny, maxx, maxy, crs


def _detect_dem_filetype(dem_folder):
    """Return the dominant raster file extension in dem_folder."""
    for ext in ['.tif', '.tiff', '.asc', '.img']:
        if glob.glob(os.path.join(dem_folder, f'*{ext}')):
            return ext
    return '.tif'


def _find_footprints_gpkg(dem_folder):
    """Return path to an existing *_footprints.gpkg in dem_folder, creating one if absent."""
    existing = glob.glob(os.path.join(dem_folder, '*_footprints.gpkg'))
    if existing:
        return existing[0]
    filetype = _detect_dem_filetype(dem_folder)
    print(f"No footprints GeoPackage found in {dem_folder}, generating one with bounding_boxes() (filetype={filetype})...", flush=True)
    gdf = _bounding_boxes(dem_folder, filetype=filetype, verbose=True)
    gpkg_files = glob.glob(os.path.join(dem_folder, '*_footprints.gpkg'))
    return gpkg_files[0]


def _select_dem_for_laz(laz_file, dem_folder, epsg=None):
    """Return the path to the DEM file in dem_folder that best covers laz_file, or None."""
    minx, miny, maxx, maxy, laz_crs = _get_laz_bounds(laz_file, epsg)
    if not laz_crs:
        print(f"Warning: {os.path.basename(laz_file)} has no CRS and no epsg provided — cannot select DEM, falling back to SMRF.")
        return None
    laz_geom = shapely_box(minx, miny, maxx, maxy)

    gpkg_path = _find_footprints_gpkg(dem_folder)
    gdf = gpd.read_file(gpkg_path)

    laz_geom_reproj = (
        gpd.GeoDataFrame(geometry=[laz_geom], crs=laz_crs).to_crs(gdf.crs).geometry.iloc[0]
    )

    intersecting = gdf[gdf.intersects(laz_geom_reproj)].copy()
    if len(intersecting) == 0:
        print(f"Warning: no DEM in {dem_folder} covers {os.path.basename(laz_file)} — falling back to SMRF.")
        return None

    intersecting['overlap_area'] = intersecting.geometry.intersection(laz_geom_reproj).area
    best_filename = intersecting.loc[intersecting['overlap_area'].idxmax(), 'filename']
    dem_path = os.path.join(dem_folder, best_filename)
    print(f"Selected DEM: {dem_path}", flush=True)
    return dem_path


def check_classified(infile, classification_code=5):
    """At least one point in the laz file has a classification_code."""
    # Quick check using the naming convention for NSW & ACT files: https://www.spatial.nsw.gov.au/__data/assets/pdf_file/0004/218992/Elevation_Data_Product_Specs.pdf
    m = re.search(r'-C(\d+)-', os.path.basename(infile)) 
    if m:
        return int(m.group(1)) >= 3
    
    # Full check if the filename didn't match the convention
    check_pipeline = {
        "pipeline": [
            infile,
            {"type": "filters.range", "limits": f"Classification[{classification_code}:{classification_code}]"},
            {"type": "filters.stats"}
        ]
    }
    p_check = pdal.Pipeline(json.dumps(check_pipeline))
    p_check.execute()
    num_points = p_check.metadata['metadata']["filters.stats"]['statistic'][0]['count'] 
    classified_bool = num_points > 0
    return classified_bool


def use_existing_classifications(infile, outdir, stub, resolution=1, classification_code=5, epsg=None, binary=False, cleanup=False):
    """Use the number of points with classification 5 (> 2m) in each pixel to generate a woody_veg tif"""

    # Some of the ACT 2015 laz files don't have an EPSG specified
    if not epsg:
        first_step = infile
    else:
        first_step = {
              "type": "readers.las",
              "filename": infile,
              "spatialreference": f'EPSG:{epsg}' # Should be "28355" for ACT 2015 LiDAR
            }

    chm_resolution = resolution
    if not binary:
        chm_resolution = 1

    # Create a raster of the number of points with the classification per pixel
    counts_tif = os.path.join(outdir, f'{stub}_counts_res{chm_resolution}_cat{classification_code}.tif')
    pipeline = {
        "pipeline": [
            first_step,
            {"type": "filters.range", "limits": f"Classification[{classification_code}:{classification_code}]"},
            {"type": "writers.gdal",
             "filename": counts_tif,
             "resolution": chm_resolution,
             "output_type": "count",
             "gdaldriver": "GTiff"},
        ]
    }
    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()
    print("Saved:", counts_tif)

    # Convert point counts into a binary raster
    counts = rxr.open_rasterio(counts_tif).isel(band=0).drop_vars('band')

    if counts.rio.crs is None:
        # Some of the ACT 2015 laz files don't have an EPSG specified
        print(f"{stub} missing crs, assuming EPSG:28355")
        counts = counts.rio.write_crs('EPSG:28355')
        counts.rio.to_raster(counts_tif)
    
    if cleanup:
        print(f"Removing: {counts_tif}")
        os.remove(counts_tif)

    num_points_threshold = 0

    da_tree = (counts > num_points_threshold).astype('uint8').rio.write_nodata(None)

    if not binary:
        # Create the percent cover tif
        percent_cover = (
            da_tree.coarsen(x=resolution, y=resolution, boundary="trim")
              .mean() * 100
        ).astype(np.uint8)
        percent_cover = percent_cover.rio.write_transform(percent_cover.rio.transform(recalc=True)) 
        tree_tif = os.path.join(outdir, f'{stub}_percentcover_res{resolution}_cat{classification_code}.tif')
        percent_cover.rio.to_raster(tree_tif)
        print("Saved:", tree_tif)
        return counts, percent_cover
    
    # Save the binary raster
    tree_tif = os.path.join(outdir, f'{stub}_woodyveg_res{resolution}_cat{classification_code}.tif')
    tif_categorical(da_tree, filename=tree_tif, colormap=cmap_woody_veg)

    return counts, da_tree


def pdal_chm(infile, outdir, stub, resolution=1, height_threshold=2, epsg=None, binary=False, cleanup=False, just_chm=False, dem=None, delineate_crowns=False, veg_only=True, uint8=False):
    """Create a canopy height model and corresponding woody_veg tif from a laz file"""

    # Some of the ACT 2015 laz files don't have an EPSG specified
    if not epsg:
        first_step = infile
    else:
        first_step = {
              "type": "readers.las",
              "filename": infile,
              "spatialreference": f'EPSG:{epsg}' # Should be "28355" for ACT 2015 LiDAR
            }

    chm_resolution = resolution
    if not binary:
        chm_resolution = 1  # Create a 1m chm, then coarsen later to the actual resolution when calculating percent cover

    # Check for vegetation classifications before building ground_steps so smrf ignores class 5 points
    veg_filter = []
    has_veg = False
    if veg_only:
        has_veg = check_classified(infile, 5)
        if has_veg:
            veg_filter = [{"type": "filters.range", "limits": "Classification[5:5]"}]  # Vegetation above 2m
            print("Found vegetation classifications (class 5), using only those points for CHM")

    if dem:
        # Providing a DEM directly is much faster than using pdal to re-calculate it.
        ground_steps = [{"type": "filters.hag_dem", "raster": dem}]
    else:
        smrf_step = {"type": "filters.smrf"}
        if has_veg:
            smrf_step["ignore"] = "Classification[5:5]"       
        ground_steps = [
            {"type": "filters.assign", "assignment": "ReturnNumber[:]=1"}, # Needed this to resolve an smrf error when NumberOfReturns or ReturnNumber = 0
            {"type": "filters.assign", "assignment": "NumberOfReturns[:]=1"},
            smrf_step,
            {"type": "filters.hag_nn"},
        ]

    # Create the canopy height tif
    chm_tif = os.path.join(outdir, f'{stub}_chm_res{chm_resolution}.tif')
    chm_json = {
        "pipeline": [
            first_step,
            *ground_steps,
            *veg_filter,
            {"type": "writers.gdal",
             "filename": chm_tif,
             "resolution": chm_resolution,
             "gdaldriver": "GTiff",
             "dimension": "HeightAboveGround",
             "output_type": "max",
             "nodata": -9999}
        ]
    }
    p_chm = pdal.Pipeline(json.dumps(chm_json))
    p_chm.execute()
    print(f"Saved: {chm_tif}", flush=True)

    if delineate_crowns:
        gdf_crowns = crowns_to_gpkg(chm_tif, outdir, stub, height_threshold)
        if gdf_crowns is not None and len(gdf_crowns) > 0:
            # Mask the CHM to only pixels inside delineated crowns so that
            # non-tree objects (powerlines, buildings) get removed.
            chm_raw = rxr.open_rasterio(chm_tif).isel(band=0).drop_vars('band')
            chm_masked = chm_raw.rio.clip(
                gdf_crowns.geometry.values,
                gdf_crowns.crs,
                drop=False,
                # all_touched=True, Any polygon touching this pixel counts. By default, the polygon has the pass through the centre of the pixel.
            ).fillna(0)
            if uint8:
                chm_masked = (chm_masked.where(chm_masked < 100, 100)
                                        .where(chm_masked != -9999, 255)
                                        .rio.write_nodata(255)
                                        .round()
                                        .astype('uint8'))
            original_chm_tif = chm_tif
            suffix = '_uint8' if uint8 else ''
            chm_tif = os.path.join(outdir, f'{stub}_chm_crowns_res{chm_resolution}{suffix}.tif')
            chm_masked.rio.to_raster(chm_tif)
            print(f"Saved: {chm_tif}", flush=True)
            os.remove(original_chm_tif)

    if just_chm:
        return None, None

    # Open the canopy height and create a binary raster
    chm = rxr.open_rasterio(chm_tif).isel(band=0).drop_vars('band')

    if chm.rio.crs is None:
        # Some of the ACT 2015 laz files don't have an EPSG specified
        print(f"{stub} missing crs, assuming EPSG:28355")
        chm = chm.rio.write_crs('EPSG:28355')
        chm.rio.to_raster(chm_tif)

    if cleanup:
        print(f"Removing: {chm_tif}")
        os.remove(chm_tif)

    # Mask nodata before thresholding since uint8 CHMs use 255 as nodata, and 255 > height_threshold
    if chm.rio.nodata is not None:
        chm = chm.where(chm != chm.rio.nodata)

    da_tree = (chm > height_threshold).astype(np.uint8).rio.write_nodata(None)

    if not binary:
        # Create the percent cover tif
        percent_cover = (
            da_tree.coarsen(x=resolution, y=resolution, boundary="trim")
              .mean() * 100
        ).astype(np.uint8)
        percent_cover = percent_cover.rio.write_transform(percent_cover.rio.transform(recalc=True))  # update the transformation, or else using rasterio to create a tif file will be 10x too small
        tree_tif = os.path.join(outdir, f'{stub}_percentcover_res{resolution}_height{height_threshold}m.tif')
        percent_cover.rio.to_raster(tree_tif)
        print(f"Saved: {tree_tif}", flush=True)
        return chm, percent_cover

    # Create the woodyveg tif
    tree_tif = os.path.join(outdir, f'{stub}_woodyveg_res{resolution}_height{height_threshold}m.tif')
    tif_categorical(da_tree, filename=tree_tif, colormap=cmap_woody_veg)

    return chm, da_tree

def lidar_folder(laz_folder, outdir='.', resolution=10, height_threshold=2, category5=False, epsg=None, binary=False, cleanup=False, just_chm=False, limit=None, dem=None, delineate_crowns=False, veg_only=True, uint8=False):
    """Apply :func:`lidar` to every .laz file in laz_folder."""
    laz_files = glob.glob(os.path.join(laz_folder,'*.laz'))
    if limit is not None:
        laz_files = laz_files[:int(limit)]
    for laz_file in laz_files:
        if os.path.getsize(laz_file) == 0:
            continue  # Some laz files from elvis are empty and this would break the pdal script
        stub = laz_file.split('/')[-1].split('.')[0]
        chm, da = lidar(laz_file, outdir, stub, resolution, height_threshold, category5, epsg, binary, cleanup, just_chm, dem=dem, delineate_crowns=delineate_crowns, veg_only=veg_only, uint8=uint8)
        del chm, da  # Trying to avoid memory accumulation
        gc.collect()


def lidar_gpkg(gpkg_file, outdir='.', resolution=10, height_threshold=2, category5=False, epsg=None, binary=False, cleanup=False, just_chm=False, limit=None, dem=None, delineate_crowns=False, veg_only=True, uint8=False, column='filepath'):
    """Apply :func:`lidar` to every row in a GeoPackage, reading laz paths from column."""
    gdf = gpd.read_file(gpkg_file)
    if limit is not None:
        gdf = gdf.iloc[:int(limit)]
    for laz_file in gdf[column]:
        if os.path.getsize(laz_file) == 0:
            continue  # Some laz files from elvis are empty and this would break the pdal script
        stub = laz_file.split('/')[-1].split('.')[0]
        chm, da = lidar(laz_file, outdir, stub, resolution, height_threshold, category5, epsg, binary, cleanup, just_chm, dem=dem, delineate_crowns=delineate_crowns, veg_only=veg_only, uint8=uint8)
        del chm, da  # Trying to avoid memory accumulation
        gc.collect()

def lidar(laz_file, outdir='.', stub='TEST', resolution=10, height_threshold=2, category5=False, epsg=None, binary=False, cleanup=False, just_chm=False, dem=None, delineate_crowns=False, veg_only=True, uint8=False):
    """
    Convert a LAZ point cloud into a canopy-height and tree-cover raster.

    If category5=True and the LAZ contains at least one point classified
    as high vegetation (LAS 1.4 category 5, see the `NSW Elevation Data Product Specification
    <https://www.spatial.nsw.gov.au/__data/assets/pdf_file/0004/218992/Elevation_Data_Product_Specs.pdf>`_),
    tree pixels are counted directly from those classified points.

    Otherwise, PDAL computes a canopy-height model from scratch 
    and thresholds it at the height_threshold. You can provide 
    a pre-computed DEM to make this process faster.
    
    The output binary raster is compatible with
    :func:`shelterbelts.indices.all_indices.indices_tif`.

    Parameters
    ----------
    laz_file : str
        Path to a .laz point cloud file.
    outdir : str, optional
        Output directory for saving results.
    stub : str, optional
        Prefix for output filenames.
    resolution : int, optional
        Pixel size in metres.
    height_threshold : float, optional
        Canopy-height cutoff in metres for the binary tree mask (only used
        when category5=False).
    category5 : bool, optional
        Use preclassified high-vegetation points (LAS category 5 meaning height > 2m) when available.
        Falls back to the PDAL CHM path if the file has no category-5 points.
    epsg : str or int, optional
        Override the LAZ file's CRS.
    binary : bool, optional
        Create just a binary tif. By default it creates a percent cover tif instead.
    cleanup : bool, optional
        Delete intermediate CHM/counts tifs after the binary raster is written.
    just_chm : bool, optional
        Only produce the canopy-height tif; skip the binary/percent-cover step.
    dem : str, optional
        Path to a DEM GeoTIFF, or a folder of DEM GeoTIFFs. When a folder is
        supplied the function selects the tile whose footprint best overlaps the
        LAZ file, auto-generating the footprints GeoPackage via
        :func:`bounding_boxes` if one does not already exist.
    delineate_crowns : bool, optional
        Run the pycrown Dalponte tree delineation and save the polygons as a gpkg.
    veg_only : bool, optional
        Restrict CHM rasterisation to points classified as high (class 5) vegetation if these exist.

    Returns
    -------
    tuple
        (chm, da_tree) — two xarray.DataArray s. chm is the raw
        canopy-height raster (or point-count raster in category5 mode);
        da_tree is the binary mask (or percent-cover mask when binary=False).

    Notes
    -----
    Writes to outdir:

    - chm.tif (if category5 = False)
    - counts.tif (if category5 = True)
    - percentcover.tif (if binary = False)
    - woodyveg.tif (if binary = True)
    - crowns.gpkg (if delineate_crowns = True)

    Examples
    --------
    Run on the bundled 50m × 50m sample LAZ from Milgadara, NSW:

    >>> from shelterbelts.utils.filepaths import laz_sample
    >>> chm, da_tree = lidar(laz_sample)
    Saved: ./TEST_chm_res1.tif
    Saved: ./TEST_percentcover_res10_height2m.tif
    """
    # Resolve a DEM folder to the specific tile that covers this LAZ file
    resolved_dem = dem
    if dem is not None and os.path.isdir(dem):
        resolved_dem = _select_dem_for_laz(laz_file, dem, epsg)

    if category5:
        # Try to use the existing classifications
        classified_bool = check_classified(laz_file, classification_code=5)  # geolocation only matters for creating the tif files later
        if classified_bool:
            classification_code = 5
            counts, da_tree = use_existing_classifications(laz_file, outdir, stub, resolution, classification_code, epsg, binary, cleanup)
            return counts, da_tree
        else:
            print("No existing classifications, generating our own canopy height model instead")

    # Do our own classifications
    chm, da_tree = pdal_chm(laz_file, outdir, stub, resolution, height_threshold, epsg, binary, cleanup, just_chm, dem=resolved_dem, delineate_crowns=delineate_crowns, veg_only=veg_only, uint8=uint8)
    return chm, da_tree


def parse_arguments():
    """Parse command line arguments for lidar() with default values."""
    parser = argparse.ArgumentParser(description="Convert a laz point cloud to a raster")

    parser.add_argument("laz_file", help="The input .laz point cloud file. If the suffix is not .laz then assume it's a folder of laz files instead.")
    parser.add_argument("--outdir", default=".", help="Output directory for saving results (default: current directory)")
    parser.add_argument("--stub", default="TEST", help="Prefix for output filenames (default: TEST)")
    parser.add_argument("--resolution", type=int, default=10, help="Pixel size in the output rasters (default: 10)")
    parser.add_argument("--height_threshold", type=float, default=2, help="Cutoff for creating the binary tif (default: 2)")
    parser.add_argument("--epsg", default=None, help="Option to specify the epsg if the .laz doesn't already have it encoded. Default: None")
    parser.add_argument("--category5", action="store_true", help="Use preclassified high vegetation (LAS 1.4 category 5). Default: False")
    parser.add_argument("--binary", action="store_true", help="Create a binary raster instead of percent tree cover. Default: False")
    parser.add_argument("--cleanup", action="store_true", help="Remove the intermediate counts or chm raster. Default: False")
    parser.add_argument("--just_chm", action="store_true", help="Don't create the binary raster. Default: False")
    parser.add_argument("--limit", default=None, help="Number of laz files to process when passing a folder. Default: None")
    parser.add_argument("--dem", default=None, help="Path to a DEM GeoTiff, or a folder of DEM GeoTiffs (the best-matching tile is selected automatically). Default: None")
    parser.add_argument("--delineate_crowns", action="store_true", help="Delineate individual tree crowns and save as a GeoPackage. Default: False")
    parser.add_argument("--no_veg_only", action="store_true", help="Use all points for the CHM even when vegetation classifications exist. Default: False")
    parser.add_argument("--uint8", action="store_true", help="Convert the CHM to uint8. Default: False")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    
    if args.laz_file.endswith('.laz'):
        lidar(
            args.laz_file,
            outdir=args.outdir,
            stub=args.stub,
            resolution=args.resolution,
            height_threshold=args.height_threshold,
            category5=args.category5,
            epsg=args.epsg,
            binary=args.binary,
            cleanup=args.cleanup,
            just_chm=args.just_chm,
            dem=args.dem,
            delineate_crowns=args.delineate_crowns,
            veg_only=not args.no_veg_only,
            uint8=args.uint8,
        )
    elif args.laz_file.endswith('.gpkg'):
        lidar_gpkg(
            args.laz_file,
            outdir=args.outdir,
            resolution=args.resolution,
            height_threshold=args.height_threshold,
            category5=args.category5,
            epsg=args.epsg,
            binary=args.binary,
            cleanup=args.cleanup,
            just_chm=args.just_chm,
            limit=args.limit,
            dem=args.dem,
            delineate_crowns=args.delineate_crowns,
            veg_only=not args.no_veg_only,
            uint8=args.uint8,
        )
    else:
        lidar_folder(
            args.laz_file,
            # We don't specify the stub, because the name of each file gets used as the stub
            outdir=args.outdir,
            resolution=args.resolution,
            height_threshold=args.height_threshold,
            category5=args.category5,
            epsg=args.epsg,
            binary=args.binary,
            cleanup=args.cleanup,
            just_chm=args.just_chm,
            limit=args.limit,
            dem=args.dem,
            delineate_crowns=args.delineate_crowns,
            veg_only=not args.no_veg_only,
            uint8=args.uint8,
        )