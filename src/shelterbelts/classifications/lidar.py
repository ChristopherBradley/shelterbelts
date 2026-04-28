# Uses PDAL bindings — requires the shelterbelts conda environment (which bundles python-pdal + rasterio).
import os
import glob
import gc
import json
import argparse

import pdal
import rioxarray as rxr
import numpy as np

from shelterbelts.utils.visualisation import tif_categorical
from shelterbelts.classifications.binary_trees import cmap_woody_veg


def check_classified(infile, classification_code=5):
    """At least one point in the laz file has a classification_code."""
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


def pdal_chm(infile, outdir, stub, resolution=1, height_threshold=2, epsg=None, binary=False, cleanup=False, just_chm=False):
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
        chm_resolution = 1  # Create a 1m chm, then coarsen to the actual resolution when calculating percent cover

    # Create the canopy height tif
    chm_tif = os.path.join(outdir, f'{stub}_chm_res{chm_resolution}.tif')
    chm_json = {
        "pipeline": [
            first_step,
            {
                "type": "filters.assign",
                "assignment": "ReturnNumber[:]=1"  # Needed this to resolve an smrf error when NumberOfReturns or ReturnNumber = 0
            },
            {
                "type": "filters.assign",
                "assignment": "NumberOfReturns[:]=1"
            },
            {"type": "filters.smrf"},  # classify ground. Should add the option for a user to provide a DEM to skip this step.
            {"type": "filters.hag_nn"},  # compute HeightAboveGround
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

    da_tree = (chm > height_threshold).astype(np.uint8).rio.write_nodata(None)  # This gives everything above the height threshold (including buildings and powerlines). Whereas using their classification code of 5 should exclude man-mdade objects.

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

def lidar_folder(laz_folder, outdir='.', resolution=10, height_threshold=2, category5=False, epsg=None, binary=False, cleanup=False, just_chm=False, limit=None):
    """Apply :func:`lidar` to every .laz file in laz_folder, skipping empty files."""
    laz_files = glob.glob(os.path.join(laz_folder,'*.laz'))
    if limit is not None:
        laz_files = laz_files[:int(limit)]
    for laz_file in laz_files:
        if os.path.getsize(laz_file) == 0:
            continue  # Some laz files from elvis are empty and this would break the pdal script
        stub = laz_file.split('/')[-1].split('.')[0]
        chm, da = lidar(laz_file, outdir, stub, resolution, height_threshold, category5, epsg, binary, cleanup, just_chm)
        del chm, da  # Trying to avoid memory accumulation
        gc.collect()

def lidar(laz_file, outdir='.', stub='TEST', resolution=10, height_threshold=2, category5=False, epsg=None, binary=False, cleanup=False, just_chm=False):
    """
    Convert a LAZ point cloud into a canopy-height and tree-cover raster.

    If category5=True and the LAZ contains at least one point classified
    as high vegetation (LAS 1.4 category 5, see the
    `NSW Elevation Data Product Specification
    <https://www.spatial.nsw.gov.au/__data/assets/pdf_file/0004/218992/Elevation_Data_Product_Specs.pdf>`_),
    tree pixels are counted directly from those classified points. 
    
    Otherwise, PDAL computes a canopy-height model
    from scratch (filters.smrf to classify ground, then filters.hag_nn
    for height-above-ground) and thresholds it at the height_threshold.

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

    Returns
    -------
    tuple
        (chm, da_tree) — two xarray.DataArray s. chm is the raw
        canopy-height raster (or point-count raster in category5 mode);
        da_tree is the binary mask (or percent-cover mask when binary=False).

    Notes
    -----
    Writes to outdir:

    - {stub}_chm_res{resolution}.tif — canopy-height model (if category5 = False)
    - {stub}_counts_res{resolution}_cat5.tif — point-count raster (if category5 = True)
    - {stub}_percentcover_res{resolution}_height{height_threshold}m.tif — (if binary = False)
    - {stub}_woodyveg_res{resolution}_height{height_threshold}m.tif — (if binary = True)

    Examples
    --------
    Run on the bundled 50m × 50m sample LAZ from Milgadara, NSW:

    >>> from shelterbelts.utils.filepaths import laz_sample
    >>> chm, da_tree = lidar(laz_sample, outdir='/tmp', stub='milgadara', resolution=5)
    Saved: /tmp/milgadara_chm_res1.tif
    Saved: /tmp/milgadara_percentcover_res5_height2m.tif
    """
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
    chm, da_tree = pdal_chm(laz_file, outdir, stub, resolution, height_threshold, epsg, binary, cleanup, just_chm)
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
            just_chm=args.just_chm
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
        )