# +
# Using a separate environment for this to my shelterbelts environment (conda activate pdal)
# # !conda install -c conda-forge pdal python-pdal rasterio
# -

import os
import glob
import pdal, json
import rioxarray as rxr
import numpy as np
import geopandas as gpd
from shapely.geometry import box

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


def use_existing_classifications(infile, outdir, stub, resolution=1, classification_code=5, epsg=None, binary=True):
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
    da_tree = (counts > 0).astype('uint8')

    if not binary:
        # Create the percent cover tif
        percent_cover = (
            da_tree.coarsen(x=resolution, y=resolution, boundary="trim")
              .mean() * 100
        ).astype(np.uint8)
        tree_tif = os.path.join(outdir, f'{stub}_percentcover_res{resolution}_cat{classification_code}.tif')
        percent_cover.rio.to_raster(tree_tif)
        print("Saved:", tree_tif)
        return counts, percent_cover
    
    # Save the binary raster
    tree_tif = os.path.join(outdir, f'{stub}_woodyveg_res{resolution}_cat{classification_code}.tif')
    tif_categorical(da_tree, filename=tree_tif, colormap=cmap_woody_veg)

    # da_tree.rio.to_raster(tree_tif)
    # print("Saved:", tree_tif)

    return counts, da_tree


def pdal_chm(infile, outdir, stub, resolution=1, height_threshold=2, epsg=None, binary=True):
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
    print("Saved:", chm_tif)

    # Open the canopy height and create a binary raster
    chm = rxr.open_rasterio(chm_tif).isel(band=0).drop_vars('band')
    da_tree = (chm > height_threshold).astype(np.uint8) # This gives everything above the height threshold, including buildings. Whereas using their classification code of 5 excludes buildings.

    if not binary:
        # Create the percent cover tif
        percent_cover = (
            da_tree.coarsen(x=resolution, y=resolution, boundary="trim")
              .mean() * 100
        ).astype(np.uint8)
        tree_tif = os.path.join(outdir, f'{stub}_percentcover_res{resolution}_height{height_threshold}m.tif')
        percent_cover.rio.to_raster(tree_tif)
        print("Saved:", tree_tif)
        return chm, percent_cover

    # Create the woodyveg tif
    tree_tif = os.path.join(outdir, f'{stub}_woodyveg_res{resolution}_height{height_threshold}m.tif')
    tif_categorical(da_tree, filename=tree_tif, colormap=cmap_woody_veg)

    # da_tree.rio.to_raster(tree_tif, compress="LZW") 
    # print("Saved:", tree_tif)

    return chm, da_tree


def tif_cleanup(outdir, size_threshold=80, percent_cover_threshold=10):
    """Remove tifs that don't meet the size or cover threshold"""
    
    # Removing all the counts, since I only care about the woody veg outputs
    counts_tifs = glob.glob(os.path.join(outdir, '*_counts_*'))
    for counts_tif in counts_tifs:
        os.remove(counts_tif)
    
    # Create a geopackage of the attributes of each tif
    veg_tifs = glob.glob(os.path.join(outdir, '*.tif'))
    records = []
    for veg_tif in veg_tifs:
        da = rxr.open_rasterio(veg_tif).isel(band=0).drop_vars("band")

        year = veg_tif.split('-')[0][-4:]
        height, width = da.shape
        bounds = da.rio.bounds()  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds
        unique, counts = np.unique(da.values, return_counts=True)
        category_counts = dict(zip(unique.tolist(), counts.tolist()))
        rec = {
            "filename": os.path.basename(veg_tif),
            "year":year,
            "height": height,
            "width": width,
            "pixels_0": category_counts.get(0, 0),
            "pixels_1": category_counts.get(1, 0),
            "geometry": box(minx, miny, maxx, maxy),
        }
        records.append(rec)
    gdf = gpd.GeoDataFrame(records, crs=da.rio.crs)
    gdf['percent_trees'] = 100 * gdf['pixels_1'] / (gdf['pixels_1'] + gdf['pixels_0']) 

    filename = os.path.join(outdir, 'tas_lidar_tif_attributes.gpkg')
    gdf.to_file(filename) # Ignore the errors from pdal, the file actually saves fine
    
    # Remove tifs that are too small, or not enough variation in trees vs no trees
    bad_tifs = gdf[(gdf['height'] < size_threshold) | (gdf['width'] < size_threshold) | 
        (gdf['percent_trees'] > 100 - percent_cover_threshold) | (gdf['percent_trees'] < percent_cover_threshold)]
    for filename in bad_tifs['filename']:
        filepath = os.path.join(outdir, filename)
        os.remove(filepath)


def lidar_folder(laz_folder, outdir='.', resolution=10, height_threshold=2, category5=False, epsg=None):
    """Run the classifications on every laz file in a folder"""
    laz_files = glob.glob(os.path.join(laz_folder,'*.laz'))
    for laz_file in laz_files:
        stub = laz_file.split('/')[-1].split('.')[0]
        lidar(laz_file, outdir, stub, resolution, height_threshold, category5, epsg)


def lidar(laz_file, outdir='.', stub='TEST', resolution=10, height_threshold=2, category5=False, epsg=None, binary=True):
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
            counts, da_tree = use_existing_classifications(laz_file, outdir, stub, resolution, classification_code, epsg, binary)
            return da_tree
        else:
            print("No existing classifications, generating our own canopy height model instead")
            
    # Do our own classifications
    chm, da_tree = pdal_chm(laz_file, outdir, stub, resolution, height_threshold, epsg, binary)
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
    parser.add_argument("--binary", action="store_true", help="Create a binary raster instead of percent tree cover. Default: False")

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
    
    lidar(
        laz_file,
        outdir=outdir,
        stub=stub,
        resolution=resolution,
        height_threshold=height_threshold,
        category5=category5,
        epsg=epsg,
        binary=binary
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
