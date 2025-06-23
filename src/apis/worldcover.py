# +
# Example code for using the planetary computer worldcover API is here: 
# https://planetarycomputer.microsoft.com/dataset/esa-worldcover#Example-Notebook
# -

# Change directory to this repo - this should work on gadi or locally via python or jupyter.
# Unfortunately, this needs to be in all files that can be run directly & use local imports.
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)

# +
# %%time
import numpy as np
import rasterio
import rioxarray # Even though this variable isn't used directly, it's needed for the da.rio methods
from pyproj import Transformer

import odc.stac
import pystac_client
import planetary_computer

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
# -

worldcover_cmap = {
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (250, 0, 0),
    60: (180, 180, 180),
    70: (240, 240, 240),
    80: (0, 100, 200),
    90: (0, 150, 160),
    95: (0, 207, 117),
    100: (250, 230, 160)
}
worldcover_labels = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss and lichen'
}


def worldcover_bbox(bbox=[147.735717, -42.912122, 147.785717, -42.862122], crs="EPSG:4326"):
    """Download worldcover data for a specific bounding box"""
    
    # Convert to EPSG:4326 because this crs is needed for the catalog search
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    bbox_4326 = [minx, miny, maxx, maxy]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox_4326
    )
    items = list(search.items())
    items = [items[0]]
    ds = odc.stac.load(items, crs="EPSG:4326", bbox=bbox_4326)
    da = ds.isel(time=0)['map']
    return da


def worldcover_centerpoint(lat=-34.3890427, lon=148.469499, buffer=0.05):
    """Download worldcover using a lat, lon & buffer"""
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    crs="EPSG:4326"
    da = worldcover_bbox(bbox, crs)
    return da


def tif_categorical(da, filename= ".", colormap=None, tiled=False):
    """Save a tif file using a categorical colour scheme"""
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=da.shape[0],
        width=da.shape[1],
        count=1,
        dtype="uint8",
        crs=da.rio.crs,
        transform=da.rio.transform(),
        compress="LZW",  # "deflate" gives slightly smaller filesize but slower
        photometric="palette",
        tiled=tiled      # Don't bother tiling unless it's a really big area (maybe bigger than 10kmx10km)
    ) as dst:
        dst.write(da.values, 1)
        if colormap:
            dst.write_colormap(1, colormap)
            
    print(f"Saved: {filename}")
    
    # If it's a really big area then you can speed up visualisation in QGIS using gdaladdo 
    # # !gdaladdo {filename_worldcover_output} 2 4 8 16 32 64


def visualise_worldcover(da):
    """Pretty visualisation using the worldcover colour scheme"""
    worldcover_classes = sorted(worldcover_cmap.keys())
    colors = [np.array(worldcover_cmap[k]) / 255.0 for k in worldcover_classes]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        boundaries=[v - 0.5 for v in worldcover_classes] + [worldcover_classes[-1] + 0.5],
        ncolors=len(worldcover_classes)
    )
    plt.figure(figsize=(8, 6))
    plt.title("ESA WorldCover")
    plt.imshow(da.values, cmap=cmap, norm=norm)
    legend_elements = [
        Patch(facecolor=np.array(color), label=worldcover_labels[class_id])
        for class_id, color in zip(worldcover_classes, colors)
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def worldcover(lat=-34.3890427, lon=148.469499, buffer=0.05, outdir=".", stub="Test"):
    """Write full documentation here"""
    max_buffer = 0.5   # This had a bug with large portions of the downloaded area being black for the qld, vic & wa coords
    if buffer > max_buffer:
        buffer = max_buffer
        print(f"Area too large, please download in smaller tiles. Reducing buffer to {max_buffer}.") 
        print(f"Estimated filesize = 10MB, estimated download time = 2 mins")
    da = worldcover_centerpoint(lat, lon, buffer)
    filename = os.path.join(outdir, f"{stub}_worldcover.tif")    
    tif_categorical(da, filename, worldcover_cmap)


# %%time
if __name__ == '__main__':
    print("Starting worldcover download")
    da = worldcover()

    # Took 5 secs, 17 secs, 3 secs (very inconsistent)

tas = -42.887122, 147.760717
anu = -35.275648, 149.100574
vic = -38.300409, 143.633974
nsw = -34.738827, 150.530493
qld = -25.817790, 149.657655
nt = -13.036965, 133.292977
wa = -32.566546, 117.523713
ocean = -35.420755, -139.774455   # Middle of nowhere in the ocean for error testing


points = [tas, anu, vic, nsw, qld, nt, wa]
stubs = ['tas', 'anu', 'vic', 'nsw', 'qld', 'nt', 'wa']

# +
# # %%time
# for point, stub in zip(points, stubs):
#     print(stub)
#     da = worldcover(point[0], point[1], 1, stub=stub)

# # Filesize is consistent, but download speed is very inconsistent
# # buffer = 0.2 -- 30 secs, 42 secs 23 MB 1.2MB, 15 secs, 28 secs, 50 secs
# # buffer = 0.3 -- 65 secs 51MB 3.5MB, 
# # buffer = 0.4 -- 180 secs
# # buffer = 0.5 -- 90 secs

# +
# If the buffer is too small than it should just get a single pixel
