import os
import glob
import rasterio
import fiona
from shapely.geometry import box, mapping, Point
from pyproj import Transformer

def bounding_boxes(folder, outdir=".", stub="TEST"):
    """Download a gpkg of bounding boxes for all the tif files in a folder"""

    tif_files = glob.glob(os.path.join(filepath, "*.tif*"))
    
    footprint_geojson = f"{outdir}/{stub}_footprints.gpkg"
    centroid_geojson = f"{outdir}/{stub}_centroids.gpkg"
    
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
    
    with fiona.open(footprint_geojson, 'w', crs=footprint_crs, schema=footprint_schema) as fp_dst, \
         fiona.open(centroid_geojson, 'w', crs=centroid_crs, schema=centroid_schema) as ct_dst:
    
        for i, tif in enumerate(tif_files):
            if i % 1000 == 0:
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
                    
    print(f"Saved: {footprint_geojson}")
    print(f"Saved: {centroid_geojson}")


if __name__ == '__main__':
    # filepath = "/Users/christopherbradley/Documents/PHD/Data/Worldcover_Australia"
    # stub = "worldcover"
    # outdir = "../../../outdir"
    # bounding_boxes(filepath, outdir, stub)

    filepath = "/scratch/xe2/cb8590/Worldcover_Australia"
    stub = "Worldcover_Australia"
    outdir = "/g/data/xe2/cb8590/Outlines"
    bounding_boxes(filepath, outdir, stub)


