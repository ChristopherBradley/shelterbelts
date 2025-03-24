# +
# NVIS data descriptions are here: https://www.dcceew.gov.au/environment/environment-information-australia/national-vegetation-information-system/data-products
# -


# !pip install matplotlib

import os
from rasterio.enums import Resampling
import numpy as np
import pyproj
import requests
import rasterio
import xarray as xr
from io import BytesIO
import rioxarray as rxr
from affine import Affine
import affine
import math
from pyproj import Transformer
import matplotlib


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

# +
# Load this canopy_height tif to grab the affine details. 
filename = os.path.join(outdir, f"{stub}_canopy_height_temp.tif")
da = rxr.open_rasterio(filename)

# Match the dimensions of the NVIS
new_y = np.linspace(float(da.y.min()), float(da.y.max()), height_pixels)
new_x = np.linspace(float(da.x.min()), float(da.x.max()), width_pixels)
resampled_da = da.interp(y=new_y, x=new_x, method="nearest")

# Remove the "band" coordina
resampled_da = resampled_da.squeeze("band").reset_coords("band", drop=True)
ds_canopy = resampled_da.to_dataset(name="canopy_height")
# -

ds_canopy

# Add the NVIS to this Dataset
new_band_da = xr.DataArray(
    ds_nvis.isel(band=0).values,
    dims=["y", "x"],
    coords={"x": ds_canopy.x, "y": ds_canopy.y},  # Band index can be adjusted
    name="NVIS"
)
ds_canopy["NVIS"] = new_band_da

ds_canopy

# Save the result as a tif
filename = os.path.join(outdir, f"{stub}_nvis_red0.tif")
ds_canopy["NVIS"].rio.to_raster(filename)
print(filename)

# +
# Assuming ds_canopy has dimensions (y, x)
y_size, x_size = ds_canopy.dims["y"], ds_canopy.dims["x"]

# Generate a random array with one band
random_band = np.random.rand(x_size, y_size)

# Create a new DataArray
new_band_da = xr.DataArray(
    random_band,
    dims=["x", "y"],
    coords={"y": ds_canopy.y, "x": ds_canopy.x},  # Band index can be adjusted
    name="random_band"
)
ds_canopy["random_band"] = new_band_da


# +
legend_url = "https://gis.environment.gov.au/gispubmap/rest/services/ogc_services/NVIS_ext_mvg/MapServer/legend?f=json"
response = requests.get(legend_url)
legend_data = response.json()

# Create a dictionary mapping category values to names and colors
category_mapping = {}
for layer in legend_data.get("layers", []):
    for item in layer.get("legend", []):
        value = int(item["label"].split()[0])  # Extract numeric ID
        name = " ".join(item["label"].split()[1:])  # Extract name
        color = tuple(item["color"])  # Extract RGBA color

        category_mapping[value] = {"name": name, "color": color}

# Print the category mapping
print(category_mapping)

# -

legend_data['layers'][0]['legend'][0]

# +
import requests
import base64
import io
from PIL import Image

# Get legend data
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
import numpy as np

# Load the raster (from Step 2)
data = ds.values  # Shape: (4, height, width) -> RGBA bands

# Convert to RGB (drop Alpha channel)
rgb_array = np.stack([data[0], data[1], data[2]], axis=-1)

# Create a categorical array
category_array = np.full(rgb_array.shape[:2], -1, dtype=np.int32)  # Default to -1 (unknown)

# Assign category values based on the closest RGB match
for rgb, category in category_mapping.items():
    mask = np.all(rgb_array == rgb, axis=-1)  # Find matching pixels
    category_array[mask] = list(category_mapping.keys()).index(rgb)  # Assign category index

print(category_array)

# +
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Convert category indices back to colors for visualization
color_array = np.zeros((*category_array.shape, 3), dtype=np.uint8)

# Track categories present in the region
unique_categories = np.unique(category_array)
legend_patches = []

for idx, (rgb, name) in enumerate(category_mapping.items()):
    if idx in unique_categories:
        color_array[category_array == idx] = rgb  # Assign legend color
        legend_patches.append(mpatches.Patch(color=np.array(rgb) / 255, label=name))

# Plot the categorical raster
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(color_array)
ax.set_title("NVIS Major Vegetation Groups")
ax.axis("off")

# Add a legend (only for categories present in the region)
if legend_patches:
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, frameon=True)

plt.show()

# -

pixel_size = 100

pixel_size=1
# minx, maxx = lon - buffer, lon + buffer
# miny, maxy = lat - buffer, lat + buffer
lon_min = minx
lat_max = maxy
