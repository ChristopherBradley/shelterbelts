# +
# Create a video of land classifications for Margaret's plots using Google Dynamic World
# Google Earth Engine Collection is here: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
# -

import ee
import geemap
import xarray as xr
import os
import rasterio
import matplotlib.pyplot as plt
import requests
import zipfile
from IPython.display import Image

ee.Authenticate()
ee.Initialize()

# Can conveniently generately these coordinates using geojson.io or bbox_finder.com
bbox_spring_valley = [149.00336491999764, -35.27196845308364, 149.02249143582833, -35.29889331119306]
bbox_bunyan_airfield = [149.124453, -36.150333, 149.157412, -36.126003]
bbox = bbox_bunyan_airfield
polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
roi = ee.Geometry.Polygon([polygon_coords])
bbox

# +
# From the DEM-H Data page, converted from javascript to python
dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H');
elevation = dataset.select('elevation');
Image(url=elevation.getThumbURL({
    'min': -10.0, 'max': 1300, 'dimensions': 512, 'region': roi,
    'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}))

asset_url = elevation.getDownloadURL({
    'scale': 30,
    'crs': 'EPSG:4326',
    'fileFormat': 'GeoTIFF',
    'region': roi})
print(asset_url)
# -

# Download the zip file
response = requests.get(asset_url, stream=True)
zip_path = 'asset.zip'
with open(zip_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=128):
        file.write(chunk)

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('asset')

# Read the TIFF file using rasterio
tif_files = [f for f in os.listdir('asset') if f.endswith('.tif')]
tif_path = os.path.join('asset', tif_files[0])
with rasterio.open(tif_path) as src:
    dem_data = src.read(1)  # Read the first band into a NumPy array

# Display the NumPy array
plt.imshow(dem_data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('DEM')
plt.show()

dem_xr.plot(cmap='terrain')

plt.imshow(dem_xr.values)

# +
# Define the dataset and region of interest
dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H')

# Convert GEE image to NumPy
array = geemap.ee_to_numpy(dataset, region=roi, bands=['elevation'], scale=30)
# -

array_2d = array[:,:,0]

# Convert NumPy array to xarray
dem_xr = xr.DataArray(array_2d, dims=('y', 'x'), name='elevation')


