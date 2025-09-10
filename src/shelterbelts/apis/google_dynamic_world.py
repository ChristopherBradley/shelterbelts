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

# # Change directory to this repo - this should work on gadi or locally via python or jupyter.
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
src_dir = os.path.join(repo_dir, 'src')
os.chdir(src_dir)
sys.path.append(src_dir)
# print(src_dir)

from shelterbelts.apis.worldcover import tif_categorical, worldcover_labels


# +
# Example GEE download of digital earth australia elevation data 
# bbox = [149.124453, -36.150333, 149.157412, -36.126003]
# polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
# roi = ee.Geometry.Polygon([polygon_coords])
# def GEE_DEA_DEM():
#     dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H')
#     array = geemap.ee_to_numpy(dataset, region=roi, bands=['elevation'], scale=30)
#     array_2d = array[:,:,0]
#     dem_xr = xr.DataArray(array_2d, dims=('y', 'x'), name='elevation')
#     dem_xr.plot(cmap='terrain')
# -

def prep_collection(bbox, start_date, end_date, image_collection="GOOGLE/DYNAMICWORLD/V1"):
    """Prep a collection of images for a given region and time range"""
    polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
    roi = ee.Geometry.Polygon([polygon_coords])
    collection = (
        ee.ImageCollection(image_collection)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .select("label")  # Select the classification band
    )
    collection_info = collection.toList(collection.size()).getInfo()
    return roi, collection_info


def download_arrays(roi, collection_info):
    """Get the values for each of the images"""
    numpy_arrays = []
    for image in collection_info:
        img = ee.Image(image["id"])
        np_array = geemap.ee_to_numpy(img, region=roi, bands=["label"], scale=10)
        numpy_arrays.append(np_array[:, :, 0])
    return numpy_arrays


def numpy_to_xarray(numpy_arrays, bbox, dates):
    """Create xarray  (I couldn't get the ee_to_xarray function to work, so this is my workaround)"""
    data_3d = np.stack(numpy_arrays, axis=0) 
    ny, nx = data_3d.shape[1:]
    x_coords = np.linspace(bbox[0], bbox[2], nx)  # longitudes
    y_coords = np.linspace(bbox[3], bbox[1], ny)  # latitudes (top → bottom)
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
    cover_da.rio.write_crs("EPSG:4326", inplace=True)
    
    # Sometimes there can be multiple values for a pixel because there are slightly overlapping sentinel tiles on the same day
    cover_unique = cover_da.groupby("time").first()
    
    # Remove timepoints where everything is water because it was probably cloud cover
    mask = (cover_unique != 0).any(dim=("y", "x"))
    cover_filtered = cover_unique.sel(time=mask)
    
    return cover_filtered


def yearly_mode(cover_da):
    """Find the most common category per year, unless it's water, then the second most common"""
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
    return cover_yearly


dynamic_world_cmap = {
 0: (65, 155, 223),
 1: (57, 125, 73),
 2: (136, 176, 83),
 3: (122, 135, 198),
 4: (228, 150, 53),
 5: (223, 195, 90),
 6: (196, 40, 27),
 7: (165, 155, 143),
 8: (179, 159, 225)
}
def tif_categorical_years(da, outdir, stub):
    """Create a tif for each year in the da"""
    years = da.time
    for year in years.values:
        filename = os.path.join(outdir, f'{stub}_{year}.tif')
        tif_categorical(da.sel(time=year), filename, dynamic_world_cmap)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
CLASS_NAMES = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice"
]
VIS_PALETTE = [
    "#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
    "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"
]
def gif_categorical(da, outdir, stub):
    """Create a gif of the dynamic world categories"""
    cmap = mcolors.ListedColormap(VIS_PALETTE)
    bounds = np.arange(len(CLASS_NAMES) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(da.isel(time=0), cmap=cmap, norm=norm)
    ax.set_title(f"Dynamic World - {da.time.values[0]}")
    cbar = plt.colorbar(im, ticks=np.arange(len(CLASS_NAMES)), ax=ax)
    cbar.set_ticklabels(CLASS_NAMES)
    
    # Update function for animation
    def update(frame):
        im.set_array(da.isel(time=frame))
        ax.set_title(f"Dynamic World - {da.time.values[frame]}")
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(da.time), interval=500)
    
    # Save or show animation
    filename = os.path.join(outdir, f'{stub}.gif')
    ani.save(filename, writer="pillow", fps=2)  # Save as GIF
    plt.show()  # Display inline if running in a Jupyter Notebook
    print("Saved:", filename)



# +
# # Maybe I can use xr_animation to make an mp4 instead of a gif?
# from dea_tools.xr_animation import xr_animation
# import matplotlib.pyplot as plt

# # Ensure da is uint8 if categorical
# da_anim = da.astype('uint8')

# # Optional: define categorical colors for plotting
# cmap = [
#     "#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
#     "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"
# ]

# # Make animation
# xr_animation(
#     da_anim,
#     filename="dynamic_world_animation.mp4",
#     cmap=cmap,
#     interval=500,      # milliseconds per frame
#     add_labels=True,   # show year on each frame
#     figsize=(6, 6)
# )
# -

def google_dynamic_world_bbox(bbox, start_date, end_date, outdir, stub):
    """Download google dynamic world categories for the bbox and time of interest"""
    rio, collection_info = prep_collection(bbox, start_date, end_date)
    dates = [c['properties']['system:index'][:8] for c in collection_info]
    numpy_arrays = download_arrays(rio, collection_info)
    da = numpy_to_xarray(numpy_arrays, bbox, dates)
    da_yearly = yearly_mode(da)
    tif_categorical_years(da_yearly, outdir, stub)
    gif_categorical(da, outdir, stub)
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
    end_date = f"{end_year}-12-31"
    da = google_dynamic_world_bbox(bbox, start_date, end_date, outdir, stub)
    return da


# +
# Took 23 secs (about 1 second per image for a 3kmx3km region)
# Took 3 mins for all timepoints from 2017 to 2025 for a 1kmx1km area

# +
import argparse

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lat', type=float, default=-34.3890427, help='Latitude in EPSG:4326 (default: -34.3890427)')
    parser.add_argument('--lon', type=float, default=148.469499, help='Longitude in EPSG:4326 (default: 148.469499)')
    parser.add_argument('--buffer', type=float, default=0.01, help='Buffer distance in degrees (~1km per 0.01 degree, default: 0.01)')
    parser.add_argument('--start_year', default='2020', help='Start year (inclusive, default: 2020)')
    parser.add_argument('--end_year', default='2021', help='End year (inclusive, default: 2021)')
    parser.add_argument('--outdir', default='.', help='Directory to save the final cleaned tiff file (default: current directory)')
    parser.add_argument('--stub', default='TEST', help='Prefix for output files (default: TEST)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    google_dynamic_world(
        lat=args.lat,
        lon=args.lon,
        buffer=args.buffer,
        start_year=args.start_year,
        end_year=args.end_year,
        outdir=args.outdir,
        stub=args.stub
    )


# +
# # Location and time of interest
# lat = -35.71799
# lon = 149.14970
# buffer = 0.005

# start_year = '2020'
# end_year = '2021'

# outdir = '/scratch/xe2/cb8590/tmp'
# stub = 'TEST_dynamic_world'

# da = google_dynamic_world(lat, lon, buffer, start_year, end_year, outdir, stub)
