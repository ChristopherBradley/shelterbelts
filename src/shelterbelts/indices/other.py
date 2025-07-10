# Find the nearest canopy height for every tree pixel in the large shelterbelts
valid_mask = ~np.isnan(ds['canopy_height'])
inds = np.array(np.nonzero(valid_mask))
dist, nearest_inds = distance_transform_edt(
    ~valid_mask,
    return_distances=True,
    return_indices=True
)
canopy_height_filled = ds['canopy_height'].values[
    tuple(nearest_inds)
]
canopy_height2 = np.full(ds['shelter'].shape, np.nan)
canopy_height2[ds['shelter'].values] = canopy_height_filled[ds['shelter'].values]

# Adjust the values because the canopy height underestimates heights in Australia
canopy_height2 = np.clip(canopy_height2 * 1.5, 5, 20)
ds['canopy_height2'] = xr.DataArray(
    canopy_height2,
    coords=ds['shelter'].coords,
    dims=ds['shelter'].dims,
    name='canopy_height2'
)
plt.imshow(canopy_height2)

# Smooth the canopy heights
valid_mask = ~np.isnan(ds['canopy_height2'])
filled = np.nan_to_num(ds['canopy_height2'].values, nan=0.0)
smoothed = gaussian_filter(filled, sigma=5)
smoothed_mask = gaussian_filter(valid_mask.astype(float), sigma=1)
smoothed_normalized = smoothed / np.where(smoothed_mask == 0, np.nan, smoothed_mask)
smoothed_canopy = np.where(ds['shelter'].values, smoothed_normalized, np.nan)
smoothed_canopy = np.clip(smoothed_canopy, 5, 20)
ds['canopy_height3'] = xr.DataArray(
    smoothed_canopy,
    coords=ds['shelter'].coords,
    dims=ds['shelter'].dims,
    name='canopy_height3'
)
ds['canopy_height3'].plot()

# Calculate the distance from nearest shelterbelt for each pixel in terms of tree heights
shelter = ds['shelter']
canopy_height = ds['canopy_height3']
wind_dir = 'W'

direction_map = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, -1),
    'W': (0, 1),
    'NE': (-1, -1),
    'NW': (-1, 1),
    'SE': (1, -1),
    'SW': (1, 1),
}
dy, dx = direction_map[wind_dir]
tree_height_distance = xr.full_like(shelter, np.nan, dtype=float)
mask = ~shelter  # only compute distances for non-tree pixels
found = shelter.copy()

pixel_size = 10  # 10m pixels
max_TH_distance = 20  # 150m for a 10m tall tree
max_pixel_distance = max_TH_distance * 2  # Assume the maximum tree height is 20m
for d in range(1, max_pixel_distance + 1):
    
    shifted_tree = found.shift(x=dx * d, y=dy * d, fill_value=False)
    shifted_height = canopy_height.shift(x=dx * d, y=dy * d)
    height_distance = (d * pixel_size) / shifted_height
    new_hits = (shifted_tree & mask) & (height_distance <= max_TH_distance)
    tree_height_distance = tree_height_distance.where(~new_hits, height_distance)
    found = found | shifted_tree
    mask = mask & ~new_hits
    if not mask.any():
        break

# Set tree pixels themselves to NaN
tree_height_distance = tree_height_distance.where(~shelter)
ds['distance_in_tree_heights'] = tree_height_distance
ds['distance_in_tree_heights'].plot()

# Calculate the distance from nearest shelterbelt for each pixel based on a set distance in metres
def compute_distance_to_tree(da, wind_dir, max_distance):
    shelter = da
    direction_map = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, -1),
        'W': (0, 1),
        'NE': (-1, -1),
        'NW': (-1, 1),
        'SE': (1, -1),
        'SW': (1, 1),
    }
    dy, dx = direction_map[wind_dir]
    distance = xr.full_like(shelter, np.nan, dtype=float)
    mask = ~shelter

    found = shelter.copy()
    pixel_size = 10  # 10m pixels
    for d in range(1, max_distance + 1):
        shifted = found.shift(x=dx, y=dy, fill_value=False)
        new_hits = shifted & mask
        distance = distance.where(~new_hits, d * pixel_size)
        found = found | shifted
        mask = mask & ~new_hits
        if not mask.any():
            break

    distances = distance.where(~shelter)
    return distances

ds['distance_to_shelterbelt'] = compute_distance_to_tree(ds['large_shelterbelts'].astype(bool), direction_20km_plus, 20)
ds['distance_to_shelterbelt'].plot()

# +
# Create a layer with the index of each category for visualisation
layers_vis = ['all_combined', 'shelter_pruned3', 'scattered_trees', 'sheltered_cropland', 'unsheltered_cropland', 'sheltered_grassland', 'unsheltered_grassland', 'Water', 'Other']
layers_legend = ['Forest', 'Shelterbelt', 'Scattered Trees', 'Sheltered Cropland', 'Unsheltered Cropland', 'Sheltered Grassland', 'Unsheltered Grassland', 'Water', 'Other']
hex_codes = ['#19670c', '#33e317', '#9b5d11', '#eca3e6', '#92688f', '#efe80d', '#ccdb73', '#3f55d2', '#8e908f']

for layer in layers_vis:
    ds[layer] = ds[layer].astype(bool)

categories = xr.zeros_like(ds[layers[0]], dtype=np.uint8)
for i, layer in enumerate(layers_vis, start=1):
    categories = categories.where(~ds[layer], other=i)  # overwrite with new category where layer is True
ds['categories'] = categories
ds['categories'].plot(cmap = 'Set1')
# -

ds['distance_in_tree_heights'].rio.to_raster('distance_in_tree_heights.tif')
ds['distance_to_shelterbelt'].rio.to_raster('distance_to_shelterbelt.tif')
ds['categories'].rio.to_raster('categories.tif')

import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.io import DatasetWriter
from matplotlib.colors import to_rgba

# +
# Apply predefined colours to the raster
with rasterio.open("categories.tif") as src:
    profile = src.profile
    data = src.read(1)

def hex_to_rgb255(hex_color):
    rgba = to_rgba(hex_color)  # values from 0-1
    return tuple(int(255 * c) for c in rgba[:3]) # values from 0-255

colormap = {
    i+1: hex_to_rgb255(hex_codes[i])
    for i in range(len(hex_codes))
}
colormap[0] = (0, 0, 0)  # optional: background color for value 0
category_metadata = {
    f"category_{i+1}": layers_legend[i]
    for i in range(len(layers_legend))
}
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress='lzw'
)
with rasterio.open("categories_colored.tif", "w", **profile) as dst:
    dst.write(data, 1)
    dst.write_colormap(1, colormap)
    dst.set_band_description(1, "Vegetation Categories")
    dst.update_tags(1, **category_metadata)
    dst.colorinterp = [ColorInterp.palette]

# -

# # Save all the layers as rasters for inspecting in QGIS
layers = list(ds.data_vars)
for layer in layers:
    filename = f"{layer}.tif"
    ds[layer].astype('uint8').rio.to_raster(filename)
    print(filename)

ds


# +
# Find pixels in the direction of the wind for a given distance from each shelterbelt
wind_rose(ds_wind, outdir="data", stub="Fulham")

# Some different ways to decide on the dominant wind direction
df, max_speed, max_direction = wind_dataframe(ds_wind)
print(df)
print(f"Maximum speed {max_speed}km/hr, Direction: {max_direction}")

# Calculating the direction with the most days with winds over 20km/hr
df_20km_plus = df.loc['20-30km/hr'] + df.loc['30+ km/hr']
direction_20km_plus = df_20km_plus.index[df_20km_plus.argmax()]
print(f"Highest percentage of days with winds > 20km/hr: {round(df_20km_plus.max(), 2)}%, Direction: {direction_20km_plus}")
# -

ds['shelter'] = ds['large_shelterbelts'].astype(bool)

# Use the canopy height to make the distances be based on tree height rather than metres
ds['shelter']

# +
# Find the nearest canopy height for every tree pixel in the large shelterbelts
valid_mask = ~np.isnan(ds['canopy_height'])
inds = np.array(np.nonzero(valid_mask))
dist, nearest_inds = distance_transform_edt(
    ~valid_mask,
    return_distances=True,
    return_indices=True
)
canopy_height_filled = ds['canopy_height'].values[
    tuple(nearest_inds)
]
canopy_height2 = np.full(ds['shelter'].shape, np.nan)
canopy_height2[ds['shelter'].values] = canopy_height_filled[ds['shelter'].values]

# Adjust the values because the canopy height underestimates heights in Australia
canopy_height2 = np.clip(canopy_height2 * 1.5, 5, 20)
ds['canopy_height2'] = xr.DataArray(
    canopy_height2,
    coords=ds['shelter'].coords,
    dims=ds['shelter'].dims,
    name='canopy_height2'
)
plt.imshow(canopy_height2)
# -

# Smooth the canopy heights
valid_mask = ~np.isnan(ds['canopy_height2'])
filled = np.nan_to_num(ds['canopy_height2'].values, nan=0.0)
smoothed = gaussian_filter(filled, sigma=5)
smoothed_mask = gaussian_filter(valid_mask.astype(float), sigma=1)
smoothed_normalized = smoothed / np.where(smoothed_mask == 0, np.nan, smoothed_mask)
smoothed_canopy = np.where(ds['shelter'].values, smoothed_normalized, np.nan)
smoothed_canopy = np.clip(smoothed_canopy, 5, 20)
ds['canopy_height3'] = xr.DataArray(
    smoothed_canopy,
    coords=ds['shelter'].coords,
    dims=ds['shelter'].dims,
    name='canopy_height3'
)
ds['canopy_height3'].plot()

# +
# Calculate the distance from nearest shelterbelt for each pixel in terms of tree heights
shelter = ds['shelter']
canopy_height = ds['canopy_height3']
wind_dir = 'W'

direction_map = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, -1),
    'W': (0, 1),
    'NE': (-1, -1),
    'NW': (-1, 1),
    'SE': (1, -1),
    'SW': (1, 1),
}
dy, dx = direction_map[wind_dir]
tree_height_distance = xr.full_like(shelter, np.nan, dtype=float)
mask = ~shelter  # only compute distances for non-tree pixels
found = shelter.copy()

pixel_size = 10  # 10m pixels
max_TH_distance = 20  # 150m for a 10m tall tree
max_pixel_distance = max_TH_distance * 2  # Assume the maximum tree height is 20m
for d in range(1, max_pixel_distance + 1):
    
    shifted_tree = found.shift(x=dx * d, y=dy * d, fill_value=False)
    shifted_height = canopy_height.shift(x=dx * d, y=dy * d)
    height_distance = (d * pixel_size) / shifted_height
    new_hits = (shifted_tree & mask) & (height_distance <= max_TH_distance)
    tree_height_distance = tree_height_distance.where(~new_hits, height_distance)
    found = found | shifted_tree
    mask = mask & ~new_hits
    if not mask.any():
        break

# Set tree pixels themselves to NaN
tree_height_distance = tree_height_distance.where(~shelter)
ds['distance_in_tree_heights'] = tree_height_distance
ds['distance_in_tree_heights'].plot()

# %%time
# Assume that the wind data has already been downloaded for this location
# buffer_degrees = 0.025
# ds_wind = barra_daily(lat=lat, lon=lon, buffer=buffer_degrees, start_year="2017", end_year="2025", outdir="../data", stub="Fulham")

filename = "data/Fulham_barra_daily.nc"
ds_wind = xr.load_dataset(filename)



# +
# Calculate the distance from nearest shelterbelt for each pixel based on a set distance in metres
def compute_distance_to_tree(da, wind_dir, max_distance):
    shelter = da
    direction_map = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, -1),
        'W': (0, 1),
        'NE': (-1, -1),
        'NW': (-1, 1),
        'SE': (1, -1),
        'SW': (1, 1),
    }
    dy, dx = direction_map[wind_dir]
    distance = xr.full_like(shelter, np.nan, dtype=float)
    mask = ~shelter

    found = shelter.copy()
    pixel_size = 10  # 10m pixels
    for d in range(1, max_distance + 1):
        shifted = found.shift(x=dx, y=dy, fill_value=False)
        new_hits = shifted & mask
        distance = distance.where(~new_hits, d * pixel_size)
        found = found | shifted
        mask = mask & ~new_hits
        if not mask.any():
            break

    distances = distance.where(~shelter)
    return distances

ds['distance_to_shelterbelt'] = compute_distance_to_tree(ds['large_shelterbelts'].astype(bool), direction_20km_plus, 20)
ds['distance_to_shelterbelt'].plot()
# -
# Create some layers for sheltered and unsheltered crop and grassland
ds['Grassland'] = (ds['worldcover'] == 30) & (~ds['all_combined'])  # Grassland and not a tree
ds['Cropland'] = (ds['worldcover'] == 40) & (~ds['all_combined'])  # Cropland and not a tree
ds['Water'] = (ds['worldcover'] == 80) & (~ds['all_combined'])  
ds['Other'] = (ds['worldcover'] != 30) & (ds['worldcover'] != 40) & (ds['worldcover'] != 80) & (~ds['all_combined'])  
ds['sheltered'] = ds['distance_to_shelterbelt'].notnull() # Within 100m of a shelterbelt in the windward direction
ds['production'] = ds['Grassland'] | ds['Cropland']
ds['unsheltered'] = ds['production'] & ~ds['sheltered']
ds['sheltered_grassland'] = (ds['sheltered'] & ds['Grassland'])
ds['sheltered_cropland'] = (ds['sheltered'] & ds['Cropland'])
ds['unsheltered_grassland'] = (ds['unsheltered'] & ds['Grassland'])
ds['unsheltered_cropland'] = (ds['unsheltered'] & ds['Cropland'])
ds['scattered_trees'] = (ds['shelterbelts'].astype(bool) & ~ds['shelter'])
ds['forest'] = ds['all_combined'] & ~ds['scattered_trees'] & ~ds['shelter_pruned3']

# Finding the inner perimeter of each group of trees that's adjacent to crop or pasture
buffers = [1,3,10]
for buffer in buffers:
# buffer = 3   # in pixels with each pixel being 10m
    diameter = buffer * 2 + 1
    production = ds['production'].values
    dilated_production = binary_dilation(production, structure=np.ones((2*buffer + 1, 2*buffer + 1)))
    shelter_near_production = shelter & dilated_production
    layer = f'shelter_pruned{buffer}'
    ds[layer] = xr.DataArray(
        shelter_near_production,
        coords=ds['shelter'].coords,
        dims=ds['shelter'].dims,
        name=layer
    )
