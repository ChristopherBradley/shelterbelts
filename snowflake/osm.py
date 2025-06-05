# +
# # !pip install osmnx
# -

import osmnx as ox
import geopandas as gpd

# +
# Location
lat = -36.370998
lon = 148.830894
buffer_deg = 0.6

# bbox
north = lat + buffer_deg
south = lat - buffer_deg
east = lon + buffer_deg
west = lon - buffer_deg
bbox = (west, south, east, north)


highway_types = ["motorway", "trunk", "primary", "secondary", "tertiary"]
# -

# %%time
roads = ox.features_from_bbox(bbox, {"highway":highway_types})

roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
roads.to_file("berridale_roads.gpkg", driver="GPKG", layer="main_roads")

roads
