# +
# Create 125 tiles, each 5kmx5km, representing similar locations to the LIDAR used to train the model in Stewart et al. 2025
# Then repeat with 125 tiles in new locations for an independent test of the model (separate from validation)
# -

import pandas as pd
import geopandas as gpd
import shapely
import json
import numpy as np
import pyproj
import shapely.geometry
from shapely.ops import transform
from shapely.geometry import Point


# +
# Define the correct projected CRS for Tasmania
crs_epsg4326 = "EPSG:4326"  # WGS 84 (lat/lon)
crs_projected = "EPSG:7855"  # GDA2020 / MGA Zone 55 (meters)

# Tile size in meters
tile_size_m = 5000  # 5 km

# Five center coordinates (WGS 84). These were chosen by eyeballing the locations in Stewart et al. 2025
training_center_coords_25km = [
    (147.6, -42.8),
    (146.9, -42.8),
    (147.4, -41.8),
    (146.7, -41.7),
    (146.3, -41.4),
]
testing_center_coords_25km = [
    (147.6, -42.3),
    (146.9, -42.3),
    (147.4, -41.3),
    (148, -41),
    (145.1, -40.9),
]
center_coords_25km = testing_center_coords_25km

# Transform function
project = pyproj.Transformer.from_crs(crs_epsg4326, crs_projected, always_xy=True).transform
unproject = pyproj.Transformer.from_crs(crs_projected, crs_epsg4326, always_xy=True).transform

# Generate 5km tiles
tiles = []
center_coords_5km = []
adj = [-2, -1, 0, 1, 2]  # 5x5 grid around each center

for lon, lat in center_coords_25km:
    # Convert center to projected coordinates
    x, y = project(lon, lat)
    
    for i in adj:
        for j in adj:
            # Calculate tile center
            x_tile = x + i * tile_size_m
            y_tile = y + j * tile_size_m
            
            # Create a 5km box around the center
            tile = shapely.geometry.box(
                x_tile - tile_size_m / 2, y_tile - tile_size_m / 2,
                x_tile + tile_size_m / 2, y_tile + tile_size_m / 2
            )
            
            # Convert back to WGS84
            tile_wgs84 = transform(unproject, tile)
            tiles.append(tile_wgs84)

            # Find the center of each tile in WGS84
            center_lon = tile_wgs84.bounds[0] + (tile_wgs84.bounds[2] - tile_wgs84.bounds[0])/2
            center_lat = tile_wgs84.bounds[1] + (tile_wgs84.bounds[3] - tile_wgs84.bounds[1])/2
            center_coords_5km.append([center_lon, center_lat])
            
# -

# Save the tiles as a geojson
grid_gdf = gpd.GeoDataFrame(geometry=tiles, crs=crs_epsg4326)
filename = "../data/tasmania_tiles_testing.geojson"
grid_gdf.to_file(filename, driver="GeoJSON")
print("Saved:", filename)

# Save the coordinates as a csv
df = pd.DataFrame(center_coords_5km, columns=["Longitude", "Latitude"])
filename = "../data/tasmania_tiles_testing.csv"
df.to_csv(filename, index=False)
print("Saved:", filename)

# Save the coordinates as a geojson to check they line up with the tile centres
geometry = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]
gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
filename_points = "../data/tasmania_center_coords_testing.geojson"
gdf_points.to_file(filename_points, driver="GeoJSON")
print("Saved points as GeoJSON:", filename_points)
