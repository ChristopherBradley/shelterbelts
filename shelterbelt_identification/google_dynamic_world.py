# +
# Create a video of land classifications for Margaret's plots using Google Dynamic World
# Google Earth Engine Collection is here: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
# -

import ee
import geemap
import xarray as xr
import matplotlib.pyplot as plt

ee.Authenticate()
ee.Initialize()

# Can conveniently generately these coordinates using geojson.io or bbox_finder.com
bbox_spring_valley = [149.00336491999764, -35.27196845308364, 149.02249143582833, -35.29889331119306]
bbox_bunyan_airfield = [149.124453, -36.150333, 149.157412, -36.126003]
bbox = bbox_bunyan_airfield
polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
roi = ee.Geometry.Polygon([polygon_coords])
bbox

# Download the data into a numpy array using 
dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H')
array = geemap.ee_to_numpy(dataset, region=roi, bands=['elevation'], scale=30)
array_2d = array[:,:,0]
dem_xr = xr.DataArray(array_2d, dims=('y', 'x'), name='elevation')

dem_xr.plot(cmap='terrain')

plt.imshow(dem_xr.values, cmap='terrain', origin='lower')
