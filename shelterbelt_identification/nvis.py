# NVIS data descriptions are here: https://www.dcceew.gov.au/environment/environment-information-australia/national-vegetation-information-system/data-products






# +
import requests
import rioxarray as rxr
import xarray as xr
from io import BytesIO

# Define the location and buffer
lat, lon = -35.0, 149.0  # Example: Canberra region
buffer = 0.05  # Buffer in degrees

# Define bounding box (WMS requires minx, miny, maxx, maxy)
minx, maxx = lon - buffer, lon + buffer
miny, maxy = lat - buffer, lat + buffer

# NVIS MapServer export URL
url = "https://gis.environment.gov.au/gispubmap/rest/services/ogc_services/NVIS_ext_mvg/MapServer/export"

# Define parameters for the request
params = {
    "bbox": f"{minx},{miny},{maxx},{maxy}",
    "bboxSR": 4326,   # Spatial reference EPSG:4326
    "imageSR": 4326,  # Output projection EPSG:4326
    "size": "512,512",  # Image resolution
    "format": "tiff",  # Request TIFF output
    "f": "image"
}

# Request the data
response = requests.get(url, params=params)
response.raise_for_status()  # Raise error for bad response

# Load response into xarray
with BytesIO(response.content) as file:
    ds = rxr.open_rasterio(file)

# Print dataset details
print(ds)

# -

ds.sel(band=1).plot()

ds.sel(band=2).plot()

ds.sel(band=3).plot()

ds.sel(band=4).plot()

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

