# The registry is here: https://registry.opendata.aws/dataforgood-fb-forests/

# +
# Standard Libraries
import os
import argparse

# Dependencies
import shutil
import numpy as np
import requests
import rasterio
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, box      

import matplotlib.pyplot as plt
from matplotlib import colors


# -

def identify_relevant_tiles_bbox(bbox=[147.735717, -42.912122, 147.785717, -42.862122], canopy_height_dir="."):
    """Find the tiles that overlap with the region of interest"""
    
    # This assumes the crs is EPSG:4326, because the aws tiles.geojson is also in EPSG:4326
    roi_coords = box(*bbox)
    roi_polygon = Polygon(roi_coords)
    
    # Download the 'tiles_global.geojson' to this folder if we haven't already
    filename = os.path.join(canopy_height_dir, 'tiles_global.geojson')
    if not os.path.exists(filename):
        url = "https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/tiles.geojson"
        with requests.get(url, stream=True) as stream:
            with open(filename, "wb") as file:
                shutil.copyfileobj(stream.raw, file)
        print(f"Downloaded {filename}", flush=True)

    # Load the canopy height tiles
    gdf = gpd.read_file(filename)

    # Find any tiles that intersect with this polygon
    relevant_tiles = []
    for idx, row in gdf.iterrows():
        tile_polygon = row['geometry']
        if tile_polygon.intersects(roi_polygon):
            relevant_tiles.append(row['tile'])
            
    return relevant_tiles


def merge_tiles_bbox(bbox, outdir=".", stub="Test", tmpdir='.'):
    """Create a tiff file with just the region of interest. This may use just one tile, or merge multiple tiles"""

    # Assumes the bbox starts as EPSG:4326
    # Convert the bounding box to EPSG:3857 (tiles.geojson uses EPSG:4326 (geographic), but the tiff files use EPSG:3857 (projected)')
    bbox_3857 = transform_bbox(bbox)
    roi_coords_3857 = box(*bbox_3857)
    roi_polygon_3857 = Polygon(roi_coords_3857)
    
    canopy_height_dir = tmpdir
    relevant_tiles = identify_relevant_tiles_bbox(bbox, canopy_height_dir)
    
    for tile in relevant_tiles:
        tiff_file = os.path.join(canopy_height_dir, f"{tile}.tif")

        # Get intersection of the tiff file and the region of interest. (any area outside this tiff file should be covered by another)
        with rasterio.open(tiff_file) as src:
            # Get bounds of the TIFF file
            tiff_bounds = src.bounds
            roi_bounds = roi_polygon_3857.bounds
            intersection_bounds = box(*tiff_bounds).intersection(box(*roi_bounds)).bounds
            window = from_bounds(*intersection_bounds, transform=src.transform)

            # Read data within the window
            out_image = src.read(window=window)
            out_transform = src.window_transform(window)
            out_meta = src.meta.copy()
    
        # Save cropped image
        cropped_tiff_filename = os.path.join(tmpdir, f"{stub}_{tile}_cropped.tif")
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    
        with rasterio.open(cropped_tiff_filename, "w", **out_meta) as dest:
            dest.write(out_image)
            
    # Merge the cropped tiffs
    src_files_to_mosaic = []
    for tile in relevant_tiles:
        tiff_file = os.path.join(tmpdir, f'{stub}_{tile}_cropped.tif')
        src = rasterio.open(tiff_file)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    for src in src_files_to_mosaic:
        src.close()

    # From visual inspection, it looks like the canopy height map is offset by about 10m south. This corrects that.
    # My hypothesis is this is due to Australia being in the southern hemisphere so shadows point south at midday, whereas the model was trained in the United States where shadows point north at midday
    original_transform = out_meta['transform']
    new_transform = original_transform * Affine.translation(0, -10)

    return mosaic, out_meta, out_trans


def identify_relevant_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, canopy_height_dir="."):
    """Find the tiles that overlap with the region of interest"""
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]  
    relevant_tiles = identify_relevant_tiles_bbox(bbox, canopy_height_dir)
    return relevant_tiles

def download_new_tiles(tiles=["311210203"], canopy_height_dir="."):
    """Download any tiles that we haven't already downloaded"""

    # Create a list of tiles we haven't downloaded yet
    to_download = []
    for tile in tiles:
        tile_path = os.path.join(canopy_height_dir, f"{tile}.tif")
        if not os.path.isfile(tile_path):
            to_download.append(tile)
    if len(to_download) == 0:
        return

    canopy_baseurl = "https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/"

    # And then download them
    print(f"Downloading {to_download}")
    for tile in to_download:
        url = canopy_baseurl + f'{tile}.tif'
        filename = os.path.join(canopy_height_dir, f'{tile}.tif')
        response = requests.head(url)
        # print(f"tile: {tile}, status_code: {response.status_code}")
        if response.status_code == 200:
            with requests.get(url, stream=True) as stream:
                with open(filename, "wb") as file:
                    shutil.copyfileobj(stream.raw, file)
            print(f"Downloaded {filename}")

def transform_bbox(bbox=[148.464499, -34.394042, 148.474499, -34.384042], inputEPSG="EPSG:4326", outputEPSG="EPSG:3857"):
    transformer = Transformer.from_crs(inputEPSG, outputEPSG)
    x1,y1 = transformer.transform(bbox[1], bbox[0])
    x2,y2 = transformer.transform(bbox[3], bbox[2])
    return (x1, y1, x2, y2)

def visualise_canopy_height(ds, filename=None):
    """Pretty visualisation of the canopy height"""

    # with rasterio.open(filename) as src:
    #     image = src.read(1)  
    
    image = ds['canopy_height']

    # Bin the slope into categories
    bin_edges = np.arange(0, 16, 1) 
    categories = np.digitize(image, bin_edges, right=True)
    
    # Define a color for each category
    colours = plt.cm.viridis(np.linspace(0, 1, len(bin_edges) - 2))
    cmap = colors.ListedColormap(['white'] + list(colours))
    
    # Plot the values
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(categories, cmap=cmap)
    
    # Assign the colours
    labels = [f'{bin_edges[i]}' for i in range(len(bin_edges))]
    labels[-1] = '>=15'
    
    # Place the tick label in the middle of each category
    num_categories = len(bin_edges)
    start_position = 0.5
    end_position = num_categories + 0.5
    step = (end_position - start_position)/(num_categories)
    tick_positions = np.arange(start_position, end_position, step)
    
    cbar = plt.colorbar(im, ticks=tick_positions)
    cbar.ax.set_yticklabels(labels)
    
    plt.title('Canopy Height (m)', size=14)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
        print("Saved:", filename)
    else:
        plt.show()


def canopy_height_bbox(bbox, outdir=".", stub="Test", tmpdir='.', save_tif=True, plot=True):
    """Create a merged canopy height raster, downloading new tiles if necessary"""
    # Assumes the bbox is in EPSG:4326
    canopy_height_dir = tmpdir
    tiles = identify_relevant_tiles_bbox(bbox, canopy_height_dir)
    download_new_tiles(tiles, canopy_height_dir)
    mosaic, out_meta, out_trans = merge_tiles_bbox(bbox, outdir, stub, tmpdir)

    # Create coordinates
    transform = out_meta['transform']
    height, width = mosaic.shape[1:]
    x = np.arange(width) * transform.a + transform.c
    y = np.arange(height) * transform.e + transform.f
    if transform.e < 0:
        y = y[::-1]
    
    # Create xarray
    da = xr.DataArray(
        mosaic,
        dims=("band", "longitude", "latitude"),
        coords={"band": ["band1"], "longitude": y, "latitude": x},
        name="canopy_height"
    ).rio.write_crs(out_meta['crs'])
    ds = da.to_dataset().squeeze('band').drop_vars(['band'])

    if save_tif:
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        output_tiff_filename = os.path.join(outdir, f'{stub}_canopy_height.tif')
        with rasterio.open(output_tiff_filename, "w", **out_meta) as dest:
            dest.write(mosaic)
        print("Saved:", output_tiff_filename)
        
    if plot:
        filename = os.path.join(outdir, f"{stub}_canopy_height.png")    
        visualise_canopy_height(ds, filename)
            
    return ds



def canopy_height(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=".", stub="Test", tmpdir='.', save_tif=True, plot=True):
    """Downlaod and create a merged canopy height raster from the Meta/Tolan dataset
    
    Parameters
    ----------
        lat, lon: Coordinates in WGS 84 (EPSG:4326).
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        outdir: The directory to save the final cleaned tiff file.
        stub: The name to be prepended to each file download.
        tmpdir: The directory to copy the original uncropped canopy height files.
        savetif: Boolean to save the final result to file.
        plot: Save a png file (not geolocated, but can be opened in Preview).

    Returns
    -------
        ds: xarray.DataSet with coords (latitude, longitude), and variable (canopy_height) of type int in metres. 
    
    Downloads
    ---------
        A Tiff file of the canopy height xarray with colours embedded.
        A png of the canopy height map including a legend.
    
    """
    minimum_buffer = 0.0001
    if buffer <= minimum_buffer:     # roughly 10m
        buffer = minimum_buffer 
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]  
    ds = canopy_height_bbox(bbox, outdir, stub, tmpdir, save_tif, plot)

    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.1', help='Buffer in each direction in degrees (default is 0.1, or about 20kmx20km)')
    parser.add_argument('--outdir', default='.', help='The directory to save the outputs. (Default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--tmpdir', default='.', help='The directory to copy the original uncropped canopy height files. (Default is the current directory)')
    parser.add_argument('--plot', default=False, action="store_true", help="Boolean to Save a png file that isn't geolocated, but can be opened in Preview. (Default: False)")

    return parser.parse_args()


# %%time
if __name__ == '__main__':

    args = parse_arguments()
    
    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    outdir = args.outdir
    tmpdir = args.tmpdir
    stub = args.stub
    plot = args.plot
    
    canopy_height(lat, lon, buffer, outdir, stub, tmpdir, plot=plot)
