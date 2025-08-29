# +
# Using a separate environment for this to my shelterbelts environment (conda activate pdal)
# # !conda install -c conda-forge pdal python-pdal rasterio
# -

import os
import pdal, json
import rioxarray
import numpy as np

from shelterbelts.util.binary_trees import tif_categorical, cmap_woody_veg # Need to remake my shelterbelts environment with pdal for this to work


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


def use_existing_classifications(infile, outdir, stub, resolution=1, classification_code=5, epsg=None):
    """Use the number of points with classification 5 (> 2m) in each pixel to generate a woody_veg tif"""

    # Create raster of points with this category per pixel 
    counts_tif = os.path.join(outdir, f'{stub}_counts_res{resolution}_cat{classification_code}.tif')
    
    # Some of the ACT 2015 laz files don't have an EPSG specified
    if not epsg:
        first_step = infile
    else:
        first_step = {
              "type": "readers.las",
              "filename": infile,
              "spatialreference": f'EPSG:{epsg}' # Should be "28355" for ACT 2015 LiDAR
            }
    
    pipeline = {
        "pipeline": [
            first_step,
            {"type": "filters.range", "limits": "Classification[5:5]"},
            {"type": "writers.gdal",
             "filename": counts_tif,
             "resolution": resolution,
             "output_type": "count",
             "gdaldriver": "GTiff"},
        ]
    }
    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()
    print("Saved:", counts_tif)

    # Convert point counts into a binary raster
    counts = rioxarray.open_rasterio(counts_tif).isel(band=0).drop_vars('band')
    da_tree = (counts > 0).astype('uint8')
    tree_tif = os.path.join(outdir, f'{stub}_woodyveg_res{resolution}_cat{classification_code}.tif')
    tif_categorical(da_tree, filename=tree_tif, colormap=cmap_woody_veg)

    # da_tree.rio.to_raster(tree_tif)
    # print("Saved:", tree_tif)

    return counts, da_tree


def pdal_chm(infile, outdir, stub, resolution=1, height_threshold=2, epsg=None):
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
    
    # Create the canopy height tif
    chm_tif = os.path.join(outdir, f'{stub}_chm_res{resolution}.tif')
    chm_json = {
        "pipeline": [
            first_step,
            {"type": "filters.smrf"},  # classify ground
            {"type": "filters.hag_nn"},  # compute HeightAboveGround
            {"type": "writers.gdal",
             "filename": chm_tif,
             "resolution": resolution,
             "gdaldriver": "GTiff",
             "dimension": "HeightAboveGround",
             "output_type": "max",
             "nodata": -9999}
        ]
    }
    p_chm = pdal.Pipeline(json.dumps(chm_json))
    p_chm.execute()
    print("Saved:", chm_tif)

    # Create the woodyveg tif
    chm = rioxarray.open_rasterio(chm_tif).isel(band=0).drop_vars('band')
    da_tree = (chm > height_threshold).astype(np.uint8) # This gives everything above the height threshold, including buildings. Whereas using their classification code of 5 excludes buildings.
    tree_tif = os.path.join(outdir, f'{stub}_woodyveg_res{resolution}_height{height_threshold}m.tif')
    tif_categorical(da_tree, filename=tree_tif, colormap=cmap_woody_veg)

    # da_tree.rio.to_raster(tree_tif, compress="LZW") 
    # print("Saved:", tree_tif)

    return chm, da_tree


def lidar(laz_file, outdir='.', stub='TEST', resolution=10, height_threshold=2, category5=False, epsg=None):
    """Convert a laz point cloud to a raster

    Parameters
    ----------
    laz_file: The .laz point cloud file
    outdir: Output directory to store the tifs
    stub: Prefix for output files
    resolution: Pixel size in the output rasters    
    height_threshold: Cutoff for creating the binary tif
    category_5: If True then it attempts to use the preclassified high vegetation from the LAS 1.4 specifications (category 5): https://www.spatial.nsw.gov.au/__data/assets/pdf_file/0004/218992/Elevation_Data_Product_Specs.pdf  

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
            outfile = os.path.join(outdir, f'{stub}_woody_veg_res{resolution}_cat5.tif')
            counts, da_tree = use_existing_classifications(laz_file, outdir, stub, resolution, epsg=epsg)
            return da_tree
        else:
            print("No existing classifications, generating our own canopy height model instead")
            
    # Do our own classifications
    chm, da_tree = pdal_chm(laz_file, outdir, stub, resolution, height_threshold, epsg=epsg)
    return da_tree


# +
import argparse

def parse_arguments():
    """Parse command line arguments for lidar() with default values."""
    parser = argparse.ArgumentParser(description="Convert a laz point cloud to a raster")

    parser.add_argument("laz_file", help="The input .laz point cloud file")
    parser.add_argument("--outdir", default=".", help="Output directory to store the tifs (default: current directory)")
    parser.add_argument("--stub", default="TEST", help="Prefix for output files (default: TEST)")
    parser.add_argument("--resolution", type=int, default=10, help="Pixel size in the output rasters (default: 10)")
    parser.add_argument("--height_threshold", type=float, default=2, help="Cutoff for creating the binary tif (default: 2)")
    parser.add_argument("--epsg", default=None, help="Option to specify the epsg if the .laz doesn't already have it encoded. Default: None")
    parser.add_argument("--category5", action="store_true", help="Use preclassified high vegetation (LAS 1.4 category 5). Default: False")

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
    
    lidar(
        laz_file,
        outdir=outdir,
        stub=stub,
        resolution=resolution,
        height_threshold=height_threshold,
        category5=category5,
        epsg=epsg
    )


# %%time
# filename = '/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz'
# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Milgadara/Point Clouds/AHD/Young201702-PHO3-C0-AHD_6306194_55_0002_0002.laz'
# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Cal/ACT Government/Point Clouds/AHD/ACT2015_4ppm-C3-AHD_6926038_55_0002_0002.laz'
# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Cal/ACT Government 2020/Point Clouds/AHD/ACT2020-12ppm-C3-AHD_6936039_55_0001_0001.laz'
# da_tree_cat5 = lidar(filename, resolution=1, category_5=True)

# +
# # %%time
# da_tree = lidar(filename, resolution=1)
