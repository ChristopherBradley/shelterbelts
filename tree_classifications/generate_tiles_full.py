# +
# Using a bounding box around Tasmania generate a list of coordinates as the center of each 6km x 6km tiles, 
# but only if the tile intersects with the Australia polygon

# Australia polygon was downloaded from here: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
# (GDA2020 Australia - 2021 - Shapefile)
# -

import pandas as pd
import geopandas as gpd
import shapely.geometry
import json
import numpy as np

# Load the Australia boundary polygon
polygon_gdf = gpd.read_file("../data/AUS_2021_AUST_SHP_GDA2020.zip")

# Bounding box surrounding Tasmania
bbox = [143, -44, 149, -39]
xmin, ymin, xmax, ymax = bbox

# Create a list for grid tiles
tiles = []
center_coords = []
tile_size = 0.05  # 5km ~ 0.05 degrees
radius = tile_size/2

# +
# %%time
# Generate grid points (tile centers)
for lon in np.arange(xmin, xmax, tile_size):
    for lat in np.arange(ymin, ymax, tile_size):
        # Create a polygon tile (square)
        tile = shapely.geometry.box(lon - radius, lat - radius, lon + radius, lat + radius)
        
        # Check if it overlaps with Australia
        if polygon_gdf.geometry.intersects(tile).any():
            center_coords.append([lat, lon])
            tiles.append(tile)

# Took 50 secs (the intersection operation is quite computationally intensive)

# +
# Save the tiles as a geojson
grid_gdf = gpd.GeoDataFrame(geometry=tiles, crs=polygon_gdf.crs)

# Round each geometry to 2dp to make the geojson more readable
def round_geometry(geom, decimals=2):
    if isinstance(geom, shapely.geometry.Polygon):
        return shapely.geometry.Polygon([(round(x, decimals), round(y, decimals)) for x, y in geom.exterior.coords])
    elif isinstance(geom, shapely.geometry.MultiPolygon):
        return shapely.geometry.MultiPolygon([
            shapely.geometry.Polygon([(round(x, decimals), round(y, decimals)) for x, y in poly.exterior.coords])
            for poly in geom.geoms
        ])
    return geom

# Apply rounding to all geometries
grid_gdf["geometry"] = grid_gdf["geometry"].apply(lambda geom: round_geometry(geom, 2))

filename = "../data/tasmania_tiles.geojson"
grid_gdf.to_file(filename, driver="GeoJSON")
print("Saved:", filename)

# +
# Save the coordinates as a csv
df = pd.DataFrame(center_coords, columns=["Latitude", "Longitude"])
df = df.round(2)

filename = "../data/tasmania_tiles.csv"
df.to_csv(filename, index=False)
print("Saved:", filename)
# -

df
