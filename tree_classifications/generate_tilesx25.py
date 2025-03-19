# +
# Using a bounding box around Tasmania generate a list of coordinates as the center of each 6km x 6km tiles, 
# but only if the tile intersects with the Australia polygon

# Australia polygon was downloaded from here: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
# (GDA2020 Australia - 2021 - Shapefile)
# -

import pandas as pd
import geopandas as gpd
import shapely
import json
import numpy as np

tile_size = 0.05  # 5km ~ 0.05 degrees
radius = tile_size/2

# +
# Five center coordinates of 25kmx25km regions that overlap with the LIDAR data available in Tasmania
c1 = 147.6, -42.7
c2 = 146.9, -42.7
c3 = 147.4, -41.8
c4 = 146.7, -41.7
c5 = 146.3, -41.4

center_coords_25km = [c1, c2, c3, c4, c5]
adj = [-2, -1, 0, 1, 2]
# -

center_coords_5km = []
tiles = []
for center_coord in center_coords_25km:
    for i in adj:
        lat = center_coord[1] + tile_size * i
        for j in adj:
            lon = center_coord[0] + tile_size * j
            lon = round(lon, 2)
            lat = round(lat, 2)
            tile = shapely.geometry.box(lon - radius, lat - radius, lon + radius, lat + radius)
            center_coords_5km.append([lat, lon])
            tiles.append(tile)

# Save the tiles as a geojson
grid_gdf = gpd.GeoDataFrame(geometry=tiles, crs="EPSG:4326")
filename = "../data/lidar_tiles_tasmania.geojson"
grid_gdf.to_file(filename, driver="GeoJSON")
print("Saved:", filename)

# Save the coordinates as a csv
df = pd.DataFrame(center_coords_5km, columns=["Latitude", "Longitude"])
df = df.round(2)
filename = "../data/lidar_tiles_tasmania.csv"
df.to_csv(filename, index=False)
print("Saved:", filename)
