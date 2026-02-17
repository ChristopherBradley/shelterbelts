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

import matplotlib.pyplot as plt
from matplotlib import colors

from shelterbelts.utils.tiles import identify_relevant_tiles_bbox, merge_tiles_bbox, transform_bbox, merged_ds

# -

def _ensure_footprints_downloaded(canopy_height_dir=".", footprints_geojson='tiles_global.geojson'):
    """Download the canopy height tile footprints GeoJSON if not already present.
    
    Parameters
    ----------
    canopy_height_dir : str
        Directory to cache the footprints file
    footprints_geojson : str
        Filename for the tile footprints
    """
    filename = os.path.join(canopy_height_dir, footprints_geojson)
    if os.path.exists(filename):
        return
    
    if footprints_geojson != 'tiles_global.geojson':
        raise ValueError('Only tiles_global.geojson is supported for auto-download')
    
    os.makedirs(canopy_height_dir, exist_ok=True)
    url = "https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/tiles.geojson"
    with requests.get(url, stream=True) as stream:
        with open(filename, "wb") as file:
            shutil.copyfileobj(stream.raw, file)
    print(f"Downloaded {filename}", flush=True)

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


def canopy_height_bbox(bbox, outdir=".", stub="Test", tmpdir='.', save_tif=True, plot=True, footprints_geojson='tiles_global.geojson'):
    """Create a merged canopy height raster, downloading new tiles if necessary"""
    # Assumes the bbox is in EPSG:4326
    canopy_height_dir = tmpdir
    
    # Ensure the tile footprints are downloaded
    _ensure_footprints_downloaded(canopy_height_dir, footprints_geojson)
    
    tiles = identify_relevant_tiles_bbox(bbox, canopy_height_dir, footprints_geojson)
    download_new_tiles(tiles, canopy_height_dir)
    mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, stub, tmpdir, footprints_geojson)

    if save_tif:
        output_tiff_filename = os.path.join(outdir, f'{stub}_canopy_height.tif')
        with rasterio.open(output_tiff_filename, "w", **out_meta) as dest:
            dest.write(mosaic)
        print("Saved:", output_tiff_filename)  # I should make a function that combines just merge_tiles_bbox and this saving
        
    ds = merged_ds(mosaic, out_meta)
    if plot:
        filename = os.path.join(outdir, f"{stub}_canopy_height.png")    
        visualise_canopy_height(ds, filename)
            
    return ds



def canopy_height(lat=-34.389, lon=148.469, buffer=0.005, outdir=".", stub="Test", tmpdir='.', save_tif=True, plot=True):
    """Download and create a merged canopy height raster from the Meta/Tolan dataset.

    Parameters
    ----------
    lat : float, optional
        Latitude in WGS 84 (EPSG:4326). Default is -34.389.
    lon : float, optional
        Longitude in WGS 84 (EPSG:4326). Default is 148.469.
    buffer : float, optional
        Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so
        a buffer of 0.01 gives an approx 2km x 2km area. Default is 0.005.
    outdir : str, optional
        Directory to save the final cleaned tiff and png. Default is the current
        directory.
    stub : str, optional
        Prefix to use for output filenames. Default is ``"Test"``.
    tmpdir : str, optional
        Directory to cache downloaded tiles. Default is the current directory.
    save_tif : bool, optional
        Whether to save the merged canopy height GeoTIFF. Default is True.
    plot : bool, optional
        Whether to save a PNG visualisation of the canopy height. Default is True.

    Returns
    -------
    xarray.Dataset
        Dataset with coordinates (latitude, longitude) and variable (canopy_height)
        of type int in metres.

    Notes
    -----
    When ``save_tif=True``, it writes:
    ``{stub}_canopy_height.tif``

    When ``plot=True``, it writes:
    ``{stub}_canopy_height.png``

    Example Visualisation
    ---------------------

    .. plot::

        import rioxarray
        from shelterbelts.apis.canopy_height import visualise_canopy_height
        from shelterbelts.utils.filepaths import get_filename

        chm_tif = get_filename('g2_26729_chm_res10_filled.tif')
        da = rioxarray.open_rasterio(chm_tif)
        ds = da.to_dataset(dim='band').rename({1: 'canopy_height'}).squeeze()
        visualise_canopy_height(ds)
    
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

    parser.add_argument('--lat', type=float, default=-34.389, help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', type=float, default=148.469, help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', type=float, default=0.005, help='Buffer in each direction in degrees (default: 0.005)')
    parser.add_argument('--outdir', default='.', help='Directory to save outputs (default: current directory)')
    parser.add_argument('--stub', default='Test', help='Prefix for output filenames (default: Test)')
    parser.add_argument('--tmpdir', default='.', help='Directory to cache downloaded tiles (default: current directory)')
    parser.add_argument('--no-save-tif', dest='save_tif', action='store_false', default=True, help='Disable saving GeoTIFF (default: enabled)')
    parser.add_argument('--no-plot', dest='plot', action='store_false', default=True, help='Disable PNG visualisation (default: enabled)')

    return parser


# %%time
if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    outdir = args.outdir
    tmpdir = args.tmpdir
    stub = args.stub
    save_tif = args.save_tif
    plot = args.plot

    canopy_height(lat, lon, buffer, outdir, stub, tmpdir, save_tif=save_tif, plot=plot)
