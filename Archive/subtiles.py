import math
import numpy as np
import geopandas as gpd

def sub_tiles(gdf_sentinel, tile_id, grid_size=5):
    """Create 25 equally spaced tiles within the larger sentinel tile"""

    polygon = gdf_sentinel.loc[gdf_sentinel['Name'] == tile_id, 'geometry'].values[0]

    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

    # 1. Automatically determine appropriate UTM zone
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    polygon_utm = gdf_utm.geometry.iloc[0]

    # 2. Compute minimum rotated rectangle and its angle
    min_rect = polygon_utm.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    angle = math.degrees(math.atan2(dy, dx))

    # 3. Rotate the polygon to axis-align it
    origin = polygon_utm.centroid
    rotated_polygon = rotate(polygon_utm, -angle, origin=origin)
    minx, miny, maxx, maxy = rotated_polygon.bounds

    # 4. Create grid inside the bounds of rotated polygon
    dx = (maxx - minx) / grid_size
    dy = (maxy - miny) / grid_size

    tiles = []
    names = []

    for i in range(grid_size):
        for j in range(grid_size):
            x0 = minx + j * dx
            y0 = miny + i * dy
            x1 = x0 + dx
            y1 = y0 + dy
            tile = box(x0, y0, x1, y1)

            # Rotate tile back to original orientation
            tile_rotated = rotate(tile, angle, origin=origin)

            tiles.append(tile_rotated)

            # Create a name from centroid
            lon, lat = gpd.GeoSeries([tile_rotated], crs=utm_crs).to_crs("EPSG:4326").centroid.iloc[0].xy
            name = f"{tile_id}{np.round(lat[0], 2)}_{np.round(lon[0], 2)}".replace(".", "_")
            names.append(name)

    # 5. Create GeoDataFrame and reproject back to EPSG:4326
    tiles_gdf = gpd.GeoDataFrame({'stub': names, 'geometry': tiles}, crs=utm_crs)
    tiles_wgs84 = tiles_gdf.to_crs("EPSG:4326")

    # 6. Save to GeoPackage
    # filename = "/scratch/xe2/cb8590/tmp/55HFC_grid_tiles.gpkg"
    filename = os.path.join(outdir_batches, f"{tile_id}.gpkg")
    if os.path.exists(filename):
        os.remove(filename)
    tiles_wgs84.to_file(filename, layer="tiles", driver="GPKG")
    print("Saved", filename)
    
    return tiles_wgs84
 
if __name__ == '__main__':
    filename_sentinel_bboxs = "/g/data/xe2/cb8590/Nick_outlines/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
    gdf_sentinel = gpd.read_file(filename_sentinel_bboxs)
    gdf = sub_tiles(gdf_sentinel, "55HFC")