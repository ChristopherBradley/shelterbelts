# The registry is here: https://registry.opendata.aws/dataforgood-fb-forests/

# +
# Standard Libraries
import os

# Dependencies
import shutil
import numpy as np
import requests
import rasterio
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
        print(f"Downloaded {filename}")

    # Load the canopy height tiles
    gdf = gpd.read_file(filename)

    # Find any tiles that intersect with this polygon
    relevant_tiles = []
    for idx, row in gdf.iterrows():
        tile_polygon = row['geometry']
        if tile_polygon.intersects(roi_polygon):
            relevant_tiles.append(row['tile'])
            
    return relevant_tiles


def merge_tiles_bbox(bbox, outdir=".", stub="Test", tmp_dir='.', canopy_height_dir='.'):
    """Create a tiff file with just the region of interest. This may use just one tile, or merge multiple tiles"""

    # Assumes the bbox starts as EPSG:4326
    # Convert the bounding box to EPSG:3857 (tiles.geojson uses EPSG:4326 (geographic), but the tiff files use EPSG:3857 (projected)')
    bbox_3857 = transform_bbox(bbox)
    roi_coords_3857 = box(*bbox_3857)
    roi_polygon_3857 = Polygon(roi_coords_3857)
    
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
        cropped_tiff_filename = os.path.join(tmp_dir, f"{stub}_{tile}_cropped.tif")
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    
        with rasterio.open(cropped_tiff_filename, "w", **out_meta) as dest:
            dest.write(out_image)
            
    # Merge the cropped tiffs
    src_files_to_mosaic = []
    for tile in relevant_tiles:
        tiff_file = os.path.join(tmp_dir, f'{stub}_{tile}_cropped.tif')
        src = rasterio.open(tiff_file)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    # From visual inspection, it looks like the canopy height map is offset by about 10m south. This corrects that.
    # My hypothesis is this is due to Australia being in the southern hemisphere so shadows point south at midday, whereas the model was trained in the United States where shadows point north at midday
    original_transform = out_meta['transform']
    new_transform = original_transform * Affine.translation(0, -10)

    # Write the merged raster to a new tiff
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    output_tiff_filename = os.path.join(outdir, f'{stub}_canopy_height.tif')
    with rasterio.open(output_tiff_filename, "w", **out_meta) as dest:
        dest.write(mosaic)
    for src in src_files_to_mosaic:
        src.close()
    print("Saved:", output_tiff_filename)
    
    return output_tiff_filename


# +
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

def merge_tiles(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=".", stub="Test", tmp_dir='.', canopy_height_dir='.'):
    """Create a tiff file with just the region of interest. This may use just one tile, or merge multiple tiles"""
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]  
    merge_tiles_bbox(bbox, outdir, stub, tmp_dir, canopy_height_dir)

def visualise_canopy_height(filename, outpath=".", stub="Test"):
    """Pretty visualisation of the canopy height"""

    with rasterio.open(filename) as src:
        image = src.read(1)  
    
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
    filename = os.path.join(outpath, f"{stub}_canopy_height.png")
    plt.savefig(filename)
    print("Saved", filename)
    plt.show()

def canopy_height(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=".", stub="Test", tmp_dir='.', canopy_height_dir="."):
    """Create a merged canopy height raster, downloading new tiles if necessary"""
    tiles = identify_relevant_tiles(lat, lon, buffer, canopy_height_dir)
    download_new_tiles(tiles, canopy_height_dir)
    merge_tiles(lat, lon, buffer, outdir, stub, tmp_dir, canopy_height_dir)


# -

def canopy_height_bbox(bbox, outdir=".", stub="Test", tmp_dir='.', canopy_height_dir="."):
    """Create a merged canopy height raster, downloading new tiles if necessary"""
    # Assumes the bbox is in EPSG:4326
    tiles = identify_relevant_tiles_bbox(bbox, canopy_height_dir)
    download_new_tiles(tiles, canopy_height_dir)
    filename = merge_tiles_bbox(bbox, outdir, stub, tmp_dir, canopy_height_dir)
    return filename


# +
# %%time
if __name__ == '__main__':

    canopy_height_dir ='../data'
    outdir = '../data'
    # canopy_height_dir ='/scratch/xe2/cb8590/Global_Canopy_Height'
    # outdir = '/scratch/xe2/cb8590/tmp'
    tmp_dir = outdir
    stub = 'Fulham'
    lat=-42.887122
    lon=147.760717
    buffer=0.04
    
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]  

    canopy_height_bbox(bbox, outdir, stub, tmp_dir, canopy_height_dir)
    filename = os.path.join(outdir, f'{stub}_canopy_height.tif')
    visualise_canopy_height(filename, outdir, stub)
    
# Took 5 mins for the download, and 33 secs for the rest
