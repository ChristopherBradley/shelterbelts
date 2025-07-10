import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr


from shelterbelts.apis.barra_daily import wind_dataframe, dominant_wind_direction
from shelterbelts.indices.tree_categories import tree_categories_labels, tree_categories_cmap
from shelterbelts.apis.worldcover import tif_categorical, visualise_categories


shelter_categories_cmap = {
    0:(255, 255, 255),
    31:(250, 242, 12),
}
shelter_categories_labels = {
    0:'Unsheltered',
    31:'Sheltered'
}
shelter_categories_labels = tree_categories_labels | shelter_categories_labels
shelter_categories_cmap = tree_categories_cmap | shelter_categories_cmap

shelter_categories_cmap

shelter_categories_labels


def compute_distance_to_tree(da, wind_dir, max_distance):
    """Calculate the distance from nearest shelterbelt for each pixel based on a set distance in metres"""
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


# Calculate the distance from nearest shelterbelt for each pixel in terms of tree heights
def bla():
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


outdir = '../../../outdir/'
stub = 'TEST'

category_tif = "../../../outdir/TEST_categorised.tif"
height_tif = "../../../outdir/TEST_canopy_height.tif"
wind_ds = "../../../outdir/TEST_barra_daily.nc"

da_categories = rxr.open_rasterio(category_tif).squeeze('band').drop_vars('band')

da_heights = rxr.open_rasterio(height_tif).squeeze('band').drop_vars('band')

ds_wind = xr.load_dataset("../../../outdir/TEST_barra_daily.nc")

wind_method = 'MOST_COMMON'
wind_threshold = 15
distance_threshold = 20

if wind_method == 'MAX':
    df, max_speed, direction_max_speed
    primary_wind_direction = direction_max_speed

if wind_method == 'MOST_COMMON':
    primary_wind_direction, df_wind = dominant_wind_direction(ds_wind, wind_threshold)

primary_wind_direction

# Counting any tree pixel that isn't a scattered tree as shelter
shelter = da_categories >= 12

distances = compute_distance_to_tree(shelter, primary_wind_direction, distance_threshold)

sheltered = distances > 0

# Assigning sheltered pixels to the label "31"
da_shelter_categories = da_categories.where(~sheltered, 31)

ds = da_shelter_categories.to_dataset(name='shelter_categories').drop_vars('spatial_ref')

# filename_png = os.path.join(outdir, f"{stub}_shelter_categories.png")
visualise_categories(ds['shelter_categories'], None, shelter_categories_cmap, shelter_categories_labels, "Shelter Categories")
        

filename = os.path.join(outdir,f"{stub}_shelter_categories.tif")
tif_categorical(ds['shelter_categories'], filename, shelter_categories_cmap)


def shelter_categories(category_tif, height_tif=None, wind_ds=None, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, savetif=True, plot=True):
    """Define sheltered and unsheltered pixels
    
    Parameters
    ----------
        category_tif: Integer tif file generated by tree_categories.py
        height_tif: Integer tif file generated by apis.canopy_height.py
            - If not provided, then sheltered/unsheltered is defined by distance in pixels rather than tree heights
        wind_ds: NetCDF with eastward and westward wind speed generated by barra_daily.py
            - If not provided, then sheltered/unsheltered is defined by nearby tree density rather than distance from trees
        wind_method: Either 'MOST_COMMON', 'MAX', or 'ALL'
            - MAX refers to the maximum wind speed
            - MOST_COMMON refers to the most common wind direction above the wind_threshold
            - ALL refers to any direction where the winds exceed the wind_threshold
        wind_threshold: Integer in km/hr
        distance_threshold: The distance from trees that counts as sheltered.
            - Units are either 'tree heights' or 'number of pixels', depending on if a height_tif is provided
        density_threshold: The percentage tree cover within the distance_threshold that counts as sheltered
            - Only applies if the wind_ds is not provided.
            
    Returns
    -------
        ds: an xarray with a band 'shelter_categories', where the integers represent the categories defined in 'shelter_category_labels'.

    Downloads
    ---------
        shelter_categories.tif: A tif file of the 'shelter_categories' band in ds, with colours embedded.
        shelter_categories.png: A png file like the tif file, but with a legend as well.
    
    """
