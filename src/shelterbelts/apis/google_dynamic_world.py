# +
# Create a video of land classifications for Margaret's plots using Google Dynamic World
# Google Earth Engine Collection is here: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

# +
# # !pip install earthengine-api

# +
# # !pip install geemap

# +
import numpy as np
import pandas as pd

import ee
import geemap
import xarray as xr
import matplotlib.pyplot as plt
# -

import geopandas as gpd
import zipfile
import shapely.wkt
import shapely.geometry
import fiona
import os

ee.Authenticate()
ee.Initialize()

# Define class names and colors
CLASS_NAMES = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice"
]
VIS_PALETTE = [
    "#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
    "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"
]

bbox = [149.124453, -36.150333, 149.157412, -36.126003]
polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
roi = ee.Geometry.Polygon([polygon_coords])
def GEE_DEA_DEM():
    # Example GEE download of digital earth australia elevation data 
    dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H')
    array = geemap.ee_to_numpy(dataset, region=roi, bands=['elevation'], scale=30)
    array_2d = array[:,:,0]
    dem_xr = xr.DataArray(array_2d, dims=('y', 'x'), name='elevation')
    dem_xr.plot(cmap='terrain')


# Can conveniently generately these coordinates using geojson.io or bbox_finder.com
bbox_spring_valley = [149.00336491999764, -35.27196845308364, 149.02249143582833, -35.29889331119306]
bbox_bunyan_airfield = [149.124453, -36.150333, 149.157412, -36.126003]
bbox = bbox_bunyan_airfield
polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
roi = ee.Geometry.Polygon([polygon_coords])
bbox

# +
lat = -35.71799
lon = 149.14970
buffer = 0.005
bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]


start_date = "2020-01-01"
end_date = "2020-03-01"
# -

bbox

# Prep a collection of images for a given region and time range
bbox = [149.1447, -35.72299, 149.1547, -35.71299]
polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
roi = ee.Geometry.Polygon([polygon_coords])
collection = (
    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    .filterBounds(roi)
    .filterDate(start_date, end_date)
    .select("label")  # Select the classification band
)
collection_info = collection.toList(collection.size()).getInfo()
dates = [c['properties']['system:index'][:8] for c in collection_info]


proj = collection.first().projection()

# %%time
# This was is working, but not automatically georeferenced
# Get the values for each of the images
numpy_arrays = []
for image in collection_info:
    img = ee.Image(image["id"])
    np_array = geemap.ee_to_numpy(img, region=roi, bands=["label"], scale=10)
    
    # Remove extra dimension (shape: (height, width, 1) → (height, width))
    numpy_arrays.append(np_array[:, :, 0])
# +
# Stack along time dimension
data_3d = np.stack(numpy_arrays, axis=0)  # Shape: (time, y, x)

# # Create xarray DataArray
# cover_da = xr.DataArray(
#     data_3d,
#     dims=("time", "y", "x"),
#     coords={"time": dates},
#     name="land_cover",
# )

# # Sometimes there can be multiple values for a pixel because there are slightly overlapping sentinel tiles on the same day
# cover_unique = cover_da.groupby("time").first()

# # Remove timepoints where everything is water because it was probably cloud cover
# mask = (cover_unique != 0).any(dim=("y", "x"))
# cover_filtered = cover_unique.sel(time=mask)


# +
# Dimensions
ny, nx = data_3d.shape[1:]  # y, x

# Generate coords
x_coords = np.linspace(bbox[0], bbox[2], nx)  # longitudes
y_coords = np.linspace(bbox[3], bbox[1], ny)  # latitudes (top → bottom)

# Create DataArray with coords
cover_da = xr.DataArray(
    data_3d,
    dims=("time", "y", "x"),
    coords={
        "time": dates,
        "x": x_coords,
        "y": y_coords,
    },
    name="land_cover",
)

# Ensure CF convention (so geospatial libs interpret correctly)
cover_da.rio.write_crs("EPSG:4326", inplace=True)

# Deduplicate by time
cover_unique = cover_da.groupby("time").first()

# Remove all-water scenes
mask = (cover_unique != 0).any(dim=("y", "x"))
cover_filtered = cover_unique.sel(time=mask)


# +
# Find the most common category per year, unless it's water, then the second most common
WATER_CLASS = 0

# 1. Convert string time to year
time_dt = pd.to_datetime(cover_da.time, format="%Y%m%d", errors='coerce')
mask_valid = ~pd.isna(time_dt)
cover_da = cover_da.sel(time=mask_valid)
years = time_dt[mask_valid].year.values

# 2. Prepare output array
unique_years = np.unique(years)
yearly_data = np.zeros((len(unique_years), cover_da.sizes['y'], cover_da.sizes['x']), dtype=cover_da.dtype)

# 3. Loop over years
for i, year in enumerate(unique_years):
    mask = years == year
    data_year = cover_da.values[mask]  # shape (time_in_year, y, x)

    # Loop over pixels (y, x)
    for y in range(data_year.shape[1]):
        for x in range(data_year.shape[2]):
            pixel_vals = data_year[:, y, x]
            counts = np.bincount(pixel_vals)
            
            # If water is present but not the only category, pick second most common
            if WATER_CLASS in np.nonzero(counts)[0] and counts.sum() > counts[WATER_CLASS]:
                counts[WATER_CLASS] = 0  # ignore water unless it's the only category
            
            yearly_data[i, y, x] = counts.argmax()

# 4. Wrap back in xarray
cover_yearly = xr.DataArray(
    yearly_data,
    dims=('time', 'y', 'x'),
    coords={'time': unique_years, 'y': cover_da.y, 'x': cover_da.x},
    name='land_cover'
)

# -

cover_yearly.rio.to_raster('/scratch/xe2/cb8590/tmp/TEST_dynamic_world.tif')

dem_xr = cover_filtered

# +
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors



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







def google_dynamic_world_bbox(bbox, start_date, end_date):
    """Download google dynamic world categories for the bbox and time of interest"""

    return da


def google_dynamic_world(lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir=".", stub="TEST"):
    """Download google dynamic world categories for the region and time of interest
    
    Parameters
    ----------
        lat, lon: Coordinates in WGS 84 (EPSG:4326).
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory to save the final cleaned tiff file.
        stub: The name to be prepended to each file download.

    Returns
    -------
        da: xarray.DataArray of the google dynamic world classifications

    Downloads
    ---------
        gdw.pkl: A pickle file of the classifications
        gdw_yearly.tif: A tif file of the most common class per year
        gdw_yearly.mp4: A video of the most common class per year
    """
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    start_date = f"{start_year}-01-01"
    end_date = f"{start_year}-12-31"
    da = google_dynamic_world_bbox(bbox, start_date, end_date)
    return da

# +
# Took 23 secs (about 1 second per image for a 3kmx3km region)
# Took 3 mins for all timepoints from 2017 to 2025 for a 1kmx1km area
