# +
# Create a geojson file showing the outline of each training sample in QGIS
# -

import os
import glob
import rasterio
from shapely.geometry import box, mapping, Point
import fiona
from fiona.crs import from_epsg
from pyproj import Transformer
import geopandas as gpd
from pathlib import Path

# +
# %%time
# Extract the bounding box and centroid for each tiff file

filepath = "/Users/christopherbradley/Documents/PHD/Data/Nick_Aus_treecover_10m"
tif_files = glob.glob(os.path.join(filepath, "*.tiff"))

footprint_geojson = "../data/tiff_footprints.geojson"
centroid_geojson = "../data/tiff_centroids.geojson"

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

with fiona.open(footprint_geojson, 'w', driver='GeoJSON', crs=footprint_crs, schema=footprint_schema) as fp_dst, \
     fiona.open(centroid_geojson, 'w', driver='GeoJSON', crs=centroid_crs, schema=centroid_schema) as ct_dst:

    for i, tif in enumerate(tif_files):
        if i % 1000 == 0:
            print(f"Working on tiff {i}/{len(tif_files)}")
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

print("GeoJSONs created:")
print(f"- {footprint_geojson}")
print(f"- {centroid_geojson}")

# +
# Next I did the intersection of LIDAR polygons & tiff polygons in QGIS to create lidar_clipped.geojson (this was too slow in python)

# Convert from geojson to gpkg for more efficient file storage
fp = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/lidar_clipped.geojson"
out_fp = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/lidar_clipped.gpkg"

gdf = gpd.read_file(fp)
gdf = gdf.rename(columns={'fid': 'fid_original', "id":'id_original'})
gdf["fid"] = range(1, len(gdf) + 1)  # Need a unique ID column for gpkg

# Save to GeoPackage
gdf.to_file(out_fp, layer='lidar_clipped', driver="GPKG")
# -

# %%time
# fp = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/lidar_clipped.geojson"
gdf2 = gpd.read_file(out_fp)
