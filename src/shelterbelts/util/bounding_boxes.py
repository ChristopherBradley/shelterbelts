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


def create_index(gpkg, tmpdir):
    """Creates a geojson from the gpkg for my tile merging function in apis.canopy_height"""
    gdf = gpd.read_file(gpkg)
    gdf['tile'] = [filename.split('.')[0] for filename in gdf['filename']]
    gdf = gdf[['tile', 'geometry']]
    gdf = gdf.to_crs('EPSG:4326')
    filename = os.path.join(tmpdir, 'tiles_global.geojson')
    gdf.to_file(filename)
    print("Saved:", filename)
    return gdf
    
# create_index('/g/data/xe2/cb8590/Outlines/Worldcover_Australia_footprints.gpkg', '/scratch/xe2/cb8590/Worldcover_Australia')
# create_index('/g/data/xe2/cb8590/Outlines/global_canopy_height_footprints.gpkg', '/scratch/xe2/cb8590/Global_Canopy_Height')


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
# filepath = '/g/data/xe2/cb8590/NSW_5m_DEMs'
# stub = 'NSW_5m_DEMs'
# outdir = "/g/data/xe2/cb8590/Outlines"
# bounding_boxes(filepath, outdir, stub, filetype='.asc')
