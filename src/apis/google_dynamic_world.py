# +
# Create a video of land classifications for Margaret's plots using Google Dynamic World
# Google Earth Engine Collection is here: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
# -

import numpy as np
import ee
import geemap
import xarray as xr
import matplotlib.pyplot as plt

import geopandas as gpd
import zipfile
import shapely.wkt
import shapely.geometry
import fiona
import os

ee.Authenticate()
ee.Initialize()

# Can conveniently generately these coordinates using geojson.io or bbox_finder.com
bbox_spring_valley = [149.00336491999764, -35.27196845308364, 149.02249143582833, -35.29889331119306]
bbox_bunyan_airfield = [149.124453, -36.150333, 149.157412, -36.126003]
bbox = bbox_bunyan_airfield
polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
roi = ee.Geometry.Polygon([polygon_coords])
bbox

# Example GEE download of digital earth australia elevation data 
dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H')
array = geemap.ee_to_numpy(dataset, region=roi, bands=['elevation'], scale=30)
array_2d = array[:,:,0]
dem_xr = xr.DataArray(array_2d, dims=('y', 'x'), name='elevation')
dem_xr.plot(cmap='terrain')

# +
# %%time
# Load the Sentinel bounding boxes
# I downloaded this file from here: https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index
filename_sentinel_bboxs = "../data/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
gdf = gpd.read_file(filename_sentinel_bboxs)
# Took 8 secs to read in all the sentinel bboxs

# Find the sentinel tile with the greatest overlap
footprint_geom = shapely.geometry.box(bbox[0], bbox[1], bbox[2], bbox[3])
gdf["overlap_area"] = gdf.geometry.intersection(footprint_geom).area
best_tile = gdf.loc[gdf["overlap_area"].idxmax()]
sentinel_tilename = best_tile['Name']
sentinel_tilename
# -

# Prep a collection of images for a given region and time range
collection = (
    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    .filterBounds(roi)
    .filterDate("2023-01-01", "2023-03-31")
    .select("label")  # Select the classification band
)

# %%time
# Get the metadata for each of the images in this filtered collection
collection_info = collection.toList(collection.size()).getInfo()
collection_info2 = [c for c in collection_info if c['id'][-5:] == sentinel_tilename]
dates = [c['properties']['system:index'][:8] for c in collection_info2]
len(dates)

# +
# %%time
# Get the values for each of the images
numpy_arrays = []
for image in collection_info:
    img = ee.Image(image["id"])
    np_array = geemap.ee_to_numpy(img, region=roi, bands=["label"], scale=10)
    
    # Remove extra dimension (shape: (height, width, 1) → (height, width))
    numpy_arrays.append(np_array[:, :, 0])

# Took 23 secs (about 1 second per image for a 3kmx3km region)

# +
# Stack along time dimension
data_3d = np.stack(numpy_arrays, axis=0)  # Shape: (time, y, x)

# Create xarray DataArray
dem_xr = xr.DataArray(
    data_3d,
    dims=("time", "y", "x"),
    coords={"time": dates},
    name="land_cover",
)
dem_xr

# +
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Define class names and colors
CLASS_NAMES = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice"
]
VIS_PALETTE = [
    "#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
    "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"
]

# Create colormap
cmap = mcolors.ListedColormap(VIS_PALETTE)
bounds = np.arange(len(CLASS_NAMES) + 1) - 0.5
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(dem_xr.isel(time=0), cmap=cmap, norm=norm)
ax.set_title(f"Dynamic World - {dem_xr.time.values[0]}")
cbar = plt.colorbar(im, ticks=np.arange(len(CLASS_NAMES)), ax=ax)
cbar.set_ticklabels(CLASS_NAMES)

# Update function for animation
def update(frame):
    im.set_array(dem_xr.isel(time=frame))
    ax.set_title(f"Dynamic World - {dem_xr.time.values[frame]}")

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(dem_xr.time), interval=500)

# Save or show animation
ani.save("dynamic_world.gif", writer="pillow", fps=2)  # Save as GIF
plt.show()  # Display inline if running in a Jupyter Notebook

# -

c = collection_info[1]

c['properties']['system:footprint']

# +



# -

bbox

# Find the sentinel tile with the greatest overlap
footprint_geom = shapely.geometry.box(bbox[0], bbox[1], bbox[2], bbox[3])
gdf["overlap_area"] = gdf.geometry.intersection(footprint_geom).area
best_tile = gdf.loc[gdf["overlap_area"].idxmax()]
sentinel_tile = best_tile['Name']
sentinel_tile

best_tile['Name']
