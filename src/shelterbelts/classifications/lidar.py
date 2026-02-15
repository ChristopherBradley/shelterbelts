# +
# Using a separate environment for this to my shelterbelts environment (conda activate pdal)
# # !conda install -c conda-forge pdal python-pdal rasterio
# -

import os
import glob
import pdal, json
import rioxarray as rxr
import numpy as np

from shelterbelts.utils.visualisation import tif_categorical
from shelterbelts.classifications.binary_trees import cmap_woody_veg # Need to remake my shelterbelts environment with pdal for this to work


def check_classified(infile, classification_code=5):
    """Check at least 1 point has this classification code"""
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


def use_existing_classifications(infile, outdir, stub, resolution=1, classification_code=5, epsg=None, binary=True, cleanup=False):
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

    da_tree = (counts > num_points_threshold).astype('uint8')

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


def pdal_chm(infile, outdir, stub, resolution=1, height_threshold=2, epsg=None, binary=True, cleanup=False, just_chm=False):
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
            {"type": "filters.smrf"},  # classify ground
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

    da_tree = (chm > height_threshold).astype(np.uint8) # This gives everything above the height threshold, including buildings. Whereas using their classification code of 5 excludes buildings.

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

import gc
def lidar_folder(laz_folder, outdir='.', resolution=10, height_threshold=2, category5=False, epsg=None, binary=True, cleanup=False, just_chm=False, limit=None):
    """Run the classifications on every laz file in a folder"""
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

def lidar(laz_file, outdir='.', stub='TEST', resolution=10, height_threshold=2, category5=False, epsg=None, binary=True, cleanup=False, just_chm=False):
    """Convert a laz point cloud to a raster

    Parameters
    ----------
    laz_file: The .laz point cloud file
    outdir: Output directory to store the tifs
    stub: Prefix for output files
    resolution: Pixel size in the output rasters    
    height_threshold: Cutoff for creating the binary tif
    category_5: If True then it attempts to use the preclassified high vegetation from the LAS 1.4 specifications (category 5): https://www.spatial.nsw.gov.au/__data/assets/pdf_file/0004/218992/Elevation_Data_Product_Specs.pdf  
    binary: If False then it generates a percent tree cover raster, instead of a binary raster

    Returns
    -------
    ds: xarray.Dataset with band 'woody_veg'
    
    Downloads
    ---------
    chm.tif
    woody_veg.tif
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


# +
import argparse

def parse_arguments():
    """Parse command line arguments for lidar() with default values."""
    parser = argparse.ArgumentParser(description="Convert a laz point cloud to a raster")

    parser.add_argument("laz_file", help="The input .laz point cloud file. If the suffix is not .laz then assume it's a folder of laz files instead.")
    parser.add_argument("--outdir", default=".", help="Output directory to store the tifs (default: current directory)")
    parser.add_argument("--stub", default="TEST", help="Prefix for output files (default: TEST)")
    parser.add_argument("--resolution", type=int, default=10, help="Pixel size in the output rasters (default: 10)")
    parser.add_argument("--height_threshold", type=float, default=2, help="Cutoff for creating the binary tif (default: 2)")
    parser.add_argument("--epsg", default=None, help="Option to specify the epsg if the .laz doesn't already have it encoded. Default: None")
    parser.add_argument("--category5", action="store_true", help="Use preclassified high vegetation (LAS 1.4 category 5). Default: False")
    parser.add_argument("--binary", action="store_true", help="Create a binary raster instead of percent tree cover. Default: False")
    parser.add_argument("--cleanup", action="store_true", help="Remove the intermediate counts or chm raster. Default: False")
    parser.add_argument("--just_chm", action="store_true", help="Don't create the binary raster. Default: False")
    parser.add_argument("--limit", default=None, help="Number of laz files to process when passing a folder. Default: None")
    return parser.parse_args()


# -

if __name__ == '__main__':

    args = parse_arguments()
    
    laz_file = args.laz_file
    outdir = args.outdir
    stub = args.stub
    resolution = args.resolution
    height_threshold = args.height_threshold
    category5 = args.category5
    epsg = args.epsg
    binary = args.binary
    cleanup = args.cleanup
    just_chm = args.just_chm
    
    if laz_file.endswith('.laz'):
        lidar(
            laz_file,
            outdir=outdir,
            stub=stub,
            resolution=resolution,
            height_threshold=height_threshold,
            category5=category5,
            epsg=epsg,
            binary=binary,
            cleanup=cleanup,
            just_chm=just_chm
        )
    else:
        lidar_folder(
            laz_file, 
            # We don't specify the stub, because the name of each file gets used as the stub
            outdir=outdir, 
            resolution=resolution, 
            height_threshold=height_threshold, 
            category5=category5, 
            epsg=epsg, 
            binary=binary,
            cleanup=cleanup,
            just_chm=just_chm,
            limit=args.limit
            )


# +
# height_threshold = 2
# resolution = 10
# filename = '../../../outdir/g2_26729_chm_res1.tif'
# chm = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

# +
# # %%time
# filename = '/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz'
# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Milgadara/Point Clouds/AHD/Young201702-PHO3-C0-AHD_6306194_55_0002_0002.laz'
# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Cal/ACT Government/Point Clouds/AHD/ACT2015_4ppm-C3-AHD_6926038_55_0002_0002.laz'
# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Cal/ACT Government 2020/Point Clouds/AHD/ACT2020-12ppm-C3-AHD_6936039_55_0001_0001.laz'
# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_09/ACT2020-12ppm-C3-AHD_6826095_55_0001_0001.laz'
# da_tree_cat5 = lidar(filename, resolution=1, category_5=True)
# da_tree = lidar(filename, resolution=1)

# +
# outdir = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Tas_tifs'

# +
# # %%time
# # laz_folder = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/TAS Government/Point Clouds/AHD/'
# # laz_folder = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/TAS_Government_2/Point Clouds/AHD' 
# laz_folder = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/TAS_Government_3/Point Clouds/AHD'

# # Took about 10 mins to process 100 tiles
# lidar_folder(laz_folder, outdir, category5=True)
# tif_cleanup(outdir)
# -