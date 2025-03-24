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


outdir = "../data"
stub = '34_0_148_5'

# +
# NVIS download with all the pixels at 100m resolution
lat, lon = -34.0, 148.5
buffer = 0.05
minx, maxx = lon - buffer, lon + buffer
miny, maxy = lat - buffer, lat + buffer
output_epsg = "EPSG:3857"  # GDA2020 / Australian Albers

# Calculate the dimensions in meters
transformer = Transformer.from_crs("EPSG:4326", output_epsg, always_xy=True)
sw_x, sw_y = transformer.transform(minx, miny)
ne_x, ne_y = transformer.transform(maxx, maxy)
width_meters = abs(ne_x - sw_x)
height_meters = abs(ne_y - sw_y)

# Calculate required pixels for 100m x 100m resolution
pixel_size = 100  # meters per pixel
width_pixels = math.ceil(width_meters / pixel_size)
height_pixels = math.ceil(height_meters / pixel_size)

# Define parameters for the request
url = "https://gis.environment.gov.au/gispubmap/rest/services/ogc_services/NVIS_ext_mvg/MapServer/export"
params = {
    "bbox": f"{minx},{miny},{maxx},{maxy}",
    "bboxSR": 4326,
    "imageSR": output_epsg,
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
# -



# +
# Download the colour labels
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

print(category_mapping)


# +
# Convert the RGB values into category indexes

# Remove the alpha channel (starts with this shape (4, height, width) representing RGBA bands)
data = ds_nvis.values  # Shape:
rgb_array = np.stack([data[0], data[1], data[2]], axis=-1)

# Assign category values
category_array = np.full(rgb_array.shape[:2], -1, dtype=np.int32)  # Default to -1 (unknown)
for rgb, category in category_mapping.items():
    mask = np.all(rgb_array == rgb, axis=-1)  # Find matching pixels
    category_array[mask] = list(category_mapping.keys()).index(rgb)  # Assign category index

print(category_array)

# +
# Georeference this NVIS category array

# Load this canopy_height tif to grab the affine details. 
filename = os.path.join(outdir, f"{stub}_canopy_height_temp.tif")
da = rxr.open_rasterio(filename)

# Change the canopy height dimensions to match the NVIS
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

# # Save the result as a tif (move this to the end)
filename = os.path.join(outdir, f"{stub}_nvis_categories.tif")
ds_canopy["NVIS"].rio.to_raster(filename)
print(filename)

# +
# Plot the categories

# Convert category indices back to colors for visualization
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

