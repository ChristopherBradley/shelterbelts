# +
import os
import argparse
import glob

import rasterio
import fiona
from shapely.geometry import box, mapping, Point
from pyproj import Transformer


# -

# Should move the api.barra_bbox function to here for consistency. 

def bounding_boxes(folder, outdir=".", stub="TEST", filetype='.tif'):
    """Download a gpkg of bounding boxes for all the tif files in a folder

    Parameters
    ----------
        folder: Folder containing lots of tifs that we want to extract the bounding box from
        outdir: The output directory to save the results 
        stub: Prefix for output file 

    Downloads
    ---------
        footprint_gpkg: A gpkg with the bounding box of each tif file and corresponding filename
        centroid_gpkg: A gpkg of the centroid of each tif file (this can be easier to view when there are lots of small tif files and you're zoomed out)
    
    """

    tif_files = glob.glob(os.path.join(filepath, f"*{filetype}*"))
    
    footprint_gpkg = f"{outdir}/{stub}_footprints.gpkg"
    centroid_gpkg = f"{outdir}/{stub}_centroids.gpkg"
    
    footprint_crs = 'EPSG:3857'
    centroid_crs = 'EPSG:4326'
    
    footprint_schema = {
        'geometry': 'Polygon',
        'properties': {'filename': 'str'}
    }
    
    centroid_schema = {
        'geometry': 'Point',
        'properties': {'filename': 'str'}
    }
    
    with fiona.open(footprint_gpkg, 'w', crs=footprint_crs, schema=footprint_schema) as fp_dst, \
         fiona.open(centroid_gpkg, 'w', crs=centroid_crs, schema=centroid_schema) as ct_dst:
    
        for i, tif in enumerate(tif_files):
            if i % 10 == 0:
                print(f"Working on tiff {i}/{len(tif_files)}")
            try:
                with rasterio.open(tif) as src:
                    bounds = src.bounds
                    src_crs = src.crs
        
                    # Transform bounds to EPSG:3857
                    footprint_transformer = Transformer.from_crs(src_crs, footprint_crs, always_xy=True)
                    minx, miny = footprint_transformer.transform(bounds.left, bounds.bottom)
                    maxx, maxy = footprint_transformer.transform(bounds.right, bounds.top)
                    geom = box(minx, miny, maxx, maxy)
        
                    # Write footprint
                    fp_dst.write({
                        'geometry': mapping(geom),
                        'properties': {'filename': os.path.basename(tif)}
                    })
        
                    # Get centroid in original CRS
                    centroid = geom.centroid
        
                    # Transform centroid to EPSG:4326
                    centroid_transformer = Transformer.from_crs(footprint_crs, centroid_crs, always_xy=True)
                    lon, lat = centroid_transformer.transform(centroid.x, centroid.y)
                    point = Point(lon, lat)
        
                    # Write centroid
                    ct_dst.write({
                        'geometry': mapping(point),
                        'properties': {'filename': os.path.basename(tif)}
                    })
            except Exception:
                    print(f"Could not open {tif}")
                    
    print(f"Saved: {footprint_gpkg}")
    print(f"Saved: {centroid_gpkg}")


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filepath', help='Folder containing lots of tifs that we want to extract the bounding box from')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', help='Prefix for output file')

    return parser.parse_args()



if __name__ == '__main__':
    
    args = parse_arguments()

    filepath = args.filepath
    outdir = args.outdir
    stub = args.stub

    bounding_boxes(filename, outdir, stub)

# +
# filepath = "/Users/christopherbradley/Documents/PHD/Data/Worldcover_Australia"
# stub = "worldcover"
# outdir = "../../../outdir"
# bounding_boxes(filepath, outdir, stub)

# filepath = "/scratch/xe2/cb8590/Worldcover_Australia"
# stub = "Worldcover_Australia"
# outdir = "/g/data/xe2/cb8590/Outlines"
# bounding_boxes(filepath, outdir, stub)

# Footprints currently aren't working with the .asc files, but centroids are for some reason.
filepath = '/g/data/xe2/cb8590/NSW_5m_DEMs'
stub = 'NSW_5m_DEMs'
outdir = "/g/data/xe2/cb8590/Outlines"
bounding_boxes(filepath, outdir, stub, filetype='.asc')

# +
# albury_dem = '/g/data/xe2/cb8590/NSW_5m_DEMs/Albury-DEM-AHD_55_5m.asc'

# with rasterio.open(albury_dem) as src:
#     bounds = src.bounds
#     src_crs = src.crs

# bounds

# import rioxarray as rxr

# da = rxr.open_rasterio(albury_dem).isel(band=0)

# da.rio.to_raster('/scratch/xe2/cb8590/tmp/albury_dem.tif')

# # !du -sh /scratch/xe2/cb8590/tmp/albury_dem.tif

# # !ls /g/data/xe2/cb8590/NSW_5m_DEMs/Albury-DEM-AHD_55_5m.asc 
