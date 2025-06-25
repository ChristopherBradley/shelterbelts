# +
# NVIS data descriptions are here: https://www.dcceew.gov.au/environment/environment-information-australia/national-vegetation-information-system/data-products


# +
import os
import io
from io import BytesIO
import base64

import math
import numpy as np
import pyproj
import requests
import rasterio
import xarray as xr
import rioxarray as rxr

import affine
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -


def nvis_rgb_raster(outdir, stub):
    """Download the rgb values for a given region from the NVIS API
    Assumes you have pre-downloaded a canopy_height tif for the bbox and affine details. 
    """
    # Load the canopy height tiff
    filename = os.path.join(outdir, f"{stub}_canopy_height.tif")
    da_canopy_height = rxr.open_rasterio(filename)
    minx,miny,maxx,maxy = da_canopy_height.rio.bounds()
    
    # Calculate required pixels for 100m x 100m resolution
    width_meters = abs(maxx - minx)
    height_meters = abs(maxy - miny)
    pixel_size = 100  # meters per pixel
    width_pixels = math.ceil(width_meters / pixel_size)
    height_pixels = math.ceil(height_meters / pixel_size)
    
    # Define parameters for the request
    url = "https://gis.environment.gov.au/gispubmap/rest/services/ogc_services/NVIS_ext_mvg/MapServer/export"
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": da_canopy_height.rio.crs.to_string(),
        "imageSR": da_canopy_height.rio.crs.to_string(),
        "size": f"{width_pixels},{height_pixels}",
        "format": "tiff",
        "f": "image"
    }
    # Request the data
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise error for bad response
    
    # Load response into xarray
    with BytesIO(response.content) as file:
        ds_nvis = rxr.open_rasterio(file)

    return ds_nvis, da_canopy_height


nvis_labels = {(146, 173, 47): 'Acacia Forests and Woodlands', (240, 228, 141): 'Acacia Open Woodlands', (250, 190, 190): 'Acacia Shrublands', (144, 186, 141): 'Callitris Forests and Woodlands', (0, 214, 168): 'Casuarina Forests and Woodlands', (252, 228, 220): 'Chenopod Shrublands, Samphire Shrublands and Forblands', (255, 255, 255): 'Cleared, non-native vegetation, buildings', (76, 230, 0): 'Eucalypt Low Open Forests', (0, 130, 0): 'Eucalypt Open Forests', (224, 255, 235): 'Eucalypt Open Woodlands', (3, 77, 0): 'Eucalypt Tall Open Forests', (193, 214, 200): 'Eucalypt Woodlands', (255, 160, 122): 'Heathlands', (255, 248, 219): 'Hummock Grasslands', (0, 11, 255): 'Inland aquatic - freshwater, salt lakes, lagoons', (138, 114, 19): 'Low Closed Forests and Tall Closed Shrublands', (224, 217, 136): 'Mallee Open Woodlands and Sparse Mallee Shrublands', (189, 182, 106): 'Mallee Woodlands and Shrublands', (21, 163, 171): 'Mangroves', (178, 235, 178): 'Melaleuca Forests and Woodlands', (204, 204, 204): 'Naturally bare - sand, rock, claypan, mudflat', (115, 255, 222): 'Other Forests and Woodlands', (252, 228, 167): 'Other Grasslands, Herblands, Sedgelands and Rushlands', (214, 157, 188): 'Other Open Woodlands', (138, 114, 101): 'Other Shrublands', (255, 0, 0): 'Rainforests and Vine Thickets', (156, 156, 156): 'Regrowth, modified native vegetation', (150, 219, 242): 'Sea and estuaries', (200, 194, 255): 'Tropical Eucalypt Woodlands/Grasslands', (184, 171, 141): 'Tussock Grasslands', (255, 170, 0): 'Unclassified Forest', (79, 79, 79): 'Unclassified native vegetation', (235, 235, 235): 'Unknown/no data'}
def download_nvis_labels():
    """Use the legend to determine the label for each colour"""
    # This should be static, so can just use the variable above instead of downloading each time
    legend_url = "https://gis.environment.gov.au/gispubmap/rest/services/ogc_services/NVIS_ext_mvg/MapServer/legend?f=json"
    response = requests.get(legend_url)
    legend_data = response.json()
    
    # Extract category mappings
    category_mapping = {}
    
    for item in legend_data["layers"][0]["legend"]:
        label = item["label"]  # Category name
        image_data = item["imageData"]  # Base64-encoded PNG
    
        # Decode Base64 image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
        # Get image size
        width, height = img.size
    
        # Sample the center pixel (avoid edges)
        center_pixel = img.getpixel((width // 2, height // 2))
    
        # Store in dictionary
        category_mapping[center_pixel] = label
        
    return category_mapping


def nvis_categories(ds, category_mapping):
    """Convert the 4 bands into a single array representing the NVIS categories"""
    
    # Remove the alpha channel (starts with this shape (4, height, width) representing RGBA bands)
    data = ds.values  # Shape:
    rgb_array = np.stack([data[0], data[1], data[2]], axis=-1)
    
    # Assign category values
    category_array = np.full(rgb_array.shape[:2], -1, dtype=np.int32)  # Default to -1 (unknown)
    for rgb, category in category_mapping.items():
        mask = np.all(rgb_array == rgb, axis=-1)  # Find matching pixels
        category_array[mask] = list(category_mapping.keys()).index(rgb)  # Assign category index

    return category_array


def georeference_nvis(category_array, da_canopy_height):
    """Add the (non-georeferenced) NVIS categories to the georeferenced canopy height xarray"""

    # Change the canopy height dimensions to match the NVIS
    da = da_canopy_height
    height_pixels, width_pixels = category_array.shape
    new_y = np.linspace(float(da.y.min()), float(da.y.max()), height_pixels)
    new_x = np.linspace(float(da.x.min()), float(da.x.max()), width_pixels)
    resampled_da = da.interp(y=new_y, x=new_x, method="nearest")
    
    # Fix the dimension names
    resampled_da = resampled_da.squeeze("band").reset_coords("band", drop=True)
    ds_canopy = resampled_da.to_dataset(name="canopy_height")
    
    # Add the NVIS to this Dataset
    new_band_da = xr.DataArray(
        category_array,
        dims=["y", "x"],
        coords={"x": ds_canopy.x, "y": ds_canopy.y},  # Band index can be adjusted
        name="NVIS"
    )
    ds_canopy["NVIS"] = new_band_da
    
    # Save the result as a tif
    filename = os.path.join(outdir, f"{stub}_nvis_categories.tif")
    ds_canopy["NVIS"].rio.to_raster(filename)
    print("Saved", filename)

    return ds_canopy


def plot_nvis(ds, category_mapping):
    """Pretty plot with category labels"""
    # Convert category indices back to colors for visualization
    category_array = ds['NVIS'].values
    color_array = np.zeros((*category_array.shape, 3), dtype=np.uint8)
    
    # Determine just the categories present in this region
    unique_categories = np.unique(category_array)
    legend_patches = []
    for idx, (rgb, name) in enumerate(category_mapping.items()):
        if idx in unique_categories:
            color_array[category_array == idx] = rgb  # Assign legend color
            legend_patches.append(mpatches.Patch(color=np.array(rgb) / 255, label=name))
    
    # Plot 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(color_array)
    ax.set_title("NVIS Major Vegetation Groups")
    ax.axis("off")
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8, frameon=True)
    plt.show()


def nvis(outdir, stub):
    """Download an NVIS tiff for the same bbox as a predownloaded canopy_height tif
    
    Assumes you have pre-downloaded a canopy_height tif for the bbox and affine details. 
    """
    ds_nvis, da_canopy_height = nvis_rgb_raster(outdir, stub)
    category_mapping = nvis_labels
    category_array = nvis_categories(ds_nvis, category_mapping)
    ds = georeference_nvis(category_array, da_canopy_height)
    return ds


if __name__ == '__main__':
    outdir = "../data"
    stub = 'Fulham'
    ds = nvis(outdir, stub)
    plot_nvis(ds, nvis_labels)
