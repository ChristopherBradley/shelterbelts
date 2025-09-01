# +
# Example code for using the planetary computer worldcover API is here: 
# https://planetarycomputer.microsoft.com/dataset/esa-worldcover#Example-Notebook


# +
# %%time
import os
import argparse

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


def tif_categorical(da, filename= "TEST.tif", colormap=None, tiled=False):
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
        tiled=tiled,      # Don't bother tiling unless it's a really big area (maybe bigger than 10kmx10km)
        nodata=da.rio.nodata 
    ) as dst:
        dst.write(da.values, 1)
        if colormap:
            dst.write_colormap(1, colormap)
            
    print(f"Saved: {filename}")
    
    # If it's a really big area then you can speed up visualisation in QGIS using gdaladdo 
    # # !gdaladdo {filename_worldcover_output} 2 4 8 16 32 64


def visualise_categories(da, filename=None, colormap=worldcover_cmap, labels=worldcover_labels, title="ESA WorldCover"):
    """Pretty visualisation using the worldcover colour scheme"""
    worldcover_classes = sorted(colormap.keys())
    
    present_classes = np.unique(da.values[~np.isnan(da.values)]).astype(int)
    worldcover_classes = [cls for cls in worldcover_classes if cls in present_classes]
    
    colors = [np.array(colormap[k]) / 255.0 for k in worldcover_classes]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        boundaries=[v - 0.5 for v in worldcover_classes] + [worldcover_classes[-1] + 0.5],
        ncolors=len(worldcover_classes)
    )
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(da.values, cmap=cmap, norm=norm)
    legend_elements = [
        Patch(facecolor=np.array(color), label=labels[class_id])
        for class_id, color in zip(worldcover_classes, colors)
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        plt.show()


def worldcover(lat=-34.3890427, lon=148.469499, buffer=0.05, outdir=".", stub="TEST", save_tif=True, plot=True):
    """Download WorldCover imagery from the microsoft planetary API

    Parameters
    ----------
        lat, lon: Coordinates in WGS 84 (EPSG:4326).
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        outdir: The directory to save the final cleaned tiff file.
        stub: The name to be prepended to each file download.
        save_tif: Boolean to determine whether to write the data to files.
        plot: Save a png file (not geolocated, but can be opened in Preview).

    Returns
    -------
        ds: xarray.DataSet with coords (latitude, longitude), and variable (worldcover) of type int. 
            The meaning of each integer is specified in worldcover_labels at the top of this file.
    
    Downloads
    ---------
        A Tiff file of the worldcover xarray with colours embedded.
        A png of the worldcover map including a legend.

    """
    print("Starting worldcover.py")

    max_buffer = 0.2   # 0.5 had a bug with large portions of the returned tif being black
    if buffer > max_buffer:
        buffer = max_buffer
        print(f"Area too large, please download in smaller tiles. Reducing buffer to {max_buffer}.") 
        print(f"Estimated filesize = 10MB, estimated download time = 2 mins")
    da = worldcover_centerpoint(lat, lon, buffer)
    ds = da.to_dataset().drop_vars(['time']).rename({'map': 'worldcover'})

    if save_tif:
        filename = os.path.join(outdir, f"{stub}_worldcover.tif")    
        tif_categorical(da, filename, worldcover_cmap)

    if plot:
        filename = os.path.join(outdir, f"{stub}_worldcover.png")    
        visualise_categories(da, filename)

    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.1', help='Buffer in each direction in degrees (default is 0.1, or about 20kmx20km)')
    parser.add_argument('--outdir', default='.', help='The directory to save the outputs. (Default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--plot', default=False, action="store_true", help="Boolean to Save a png file that isn't geolocated, but can be opened in Preview. (Default: False)")

    return parser.parse_args()


# %%time
if __name__ == '__main__':
    args = parse_arguments()
    
    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    outdir = args.outdir
    stub = args.stub
    plot = args.plot
    
    worldcover(lat, lon, buffer, outdir, stub, plot=plot)