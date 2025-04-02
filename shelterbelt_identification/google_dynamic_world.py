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

# Download the data into a numpy array using geemap
dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H')
array = geemap.ee_to_numpy(dataset, region=roi, bands=['elevation'], scale=30)
array_2d = array[:,:,0]
dem_xr = xr.DataArray(array_2d, dims=('y', 'x'), name='elevation')

dem_xr.plot(cmap='terrain')

# Following the example to download dynamic earth: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1#colab-python


START = ee.Date('2021-04-02')
END = START.advance(1, 'day')

# Filter by region and date
col_filter = ee.Filter.And(
    ee.Filter.bounds(ee.Geometry.Point(20.6729, 52.4305)),
    ee.Filter.date(START, END),
)

# Prep the imagecollections
dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(col_filter)
s2_col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED');


# What does linking do?
linked_col = dw_col.linkCollection(s2_col, s2_col.first().bandNames());

linked_image = ee.Image(linked_col.first())

# +
# Prep the colours
CLASS_NAMES = [
    'water',
    'trees',
    'grass',
    'flooded_vegetation',
    'crops',
    'shrub_and_scrub',
    'built',
    'bare',
    'snow_and_ice',
]

VIS_PALETTE = [
    '419bdf',
    '397d49',
    '88b053',
    '7a87c6',
    'e49635',
    'dfc35a',
    'c4281b',
    'a59b8f',
    'b39fe1',
]
# -

# Create an RGB image of the label (most likely class) on [0, 1].
dw_rgb = (
    linked_image.select('label')
    .visualize(min=0, max=8, palette=VIS_PALETTE)
    .divide(255)
)

# Get the most likely class probability.
top1_prob = linked_image.select(CLASS_NAMES).reduce(ee.Reducer.max())


# Create a hillshade of the most likely class probability on [0, 1]
top1_prob_hillshade = ee.Terrain.hillshade(top1_prob.multiply(100)).divide(255)


# Combine the RGB image with the hillshade.
dw_rgb_hillshade = dw_rgb.multiply(top1_prob_hillshade)


# Display the Dynamic World visualization with the source Sentinel-2 image.
m = geemap.Map()
m.set_center(20.6729, 52.4305, 12)
m.add_layer(
    linked_image,
    {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']},
    'Sentinel-2 L1C',
)
m.add_layer(
    dw_rgb_hillshade,
    {'min': 0, 'max': 0.65},
    'Dynamic World V1 - label hillshade',
)
m
