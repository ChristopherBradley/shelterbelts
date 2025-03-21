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
