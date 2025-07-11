import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import rasterio as rio
import scipy


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
def compute_distance_to_tree_TH(shelter_heights, wind_dir='E', max_distance=20, multi_heights=True):
    """Compute the distance from each pixel.
       shelter_heights is an array of tree heights, with non-trees being np.nan.
    """
    distance_threshold = max_distance
    dy, dx = direction_map[wind_dir]
    
    # Finding the height of the edge sheltering each pixel
    shifted = shelter_heights.copy()
    shifted_full_max = shelter_heights.copy()
    for d in range(1, distance_threshold + 1):
        shifted = shifted.shift(x=dx, y=dy, fill_value=np.nan)
        shifted = shifted.where(shifted_full_max.isnull(), np.nan)
        shifted = shifted.where(shifted > 0, np.nan)
        shifted_full_max = shifted_full_max.where(~shifted_full_max.isnull(), shifted)
    
    # Finding the distance from the edge sheltering each pixel
    shifted = shelter_heights.copy()
    shifted_full = shelter_heights.copy()
    for d in range(1, distance_threshold + 1):
        shifted = shifted.shift(x=dx, y=dy, fill_value=np.nan)
        shifted = shifted.where(shifted_full.isnull(), np.nan)
        shifted = shifted - 1
        shifted = shifted.where(shifted > 0, np.nan)
        shifted_full = shifted_full.where(~shifted_full.isnull(), shifted)
    
    new_hits = shifted_full.where(~shelter, np.nan) 
    distances = (shifted_full_max - new_hits)

    # If multi_heights is False, than we assume only the edge trees can provide shelter.
    # Otherwise, we also incorporate trees inside the edge if they're tall enough to provide shelter when the edge trees are not.
    if multi_heights:
        # Finding the height of the centre tree sheltering each pixel
        shifted = shelter_heights.copy()
        shifted_full = shelter_heights.copy()
        for d in range(1, distance_threshold + 1):
            shifted = shifted.shift(x=dx, y=dy, fill_value=np.nan)
            shifted = shifted.where(shifted > 0, np.nan)
            shifted_full_max_centre = shifted_full.where(~shifted_full.isnull(), shifted)
        
        # Finding the distance from the centre tree sheltering each pixel
        shifted = shelter_heights.copy()
        shifted_full = shelter_heights.copy()
        for d in range(1, distance_threshold + 1):
            shifted = shifted.shift(x=dx, y=dy, fill_value=np.nan)
            shifted = shifted - 1
            shifted = shifted.where(shifted > 0, np.nan)
            shifted_full_centre = shifted_full.where(~shifted_full.isnull(), shifted)
        
        new_hits = shifted_full_centre.where(~shelter, np.nan) 
        distances_centre = (shifted_full_max_centre - new_hits)
        
        distances = distances.where(~distances.isnull(), distances_centre)
    
    return distances


def computer_tree_densities(tree_percent, min_distance=0, max_distance=20):
    """Using a boolean tree mask, assign a percentage density to each pixel within the given distance"""
    tree_mask = tree_percent > 0
    
    structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
    adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)
        
    # Calculate the number of trees in a donut between the inner and outer circle
    y, x = np.ogrid[-max_distance:max_distance+1, -max_distance:max_distance+1]
    kernel = (x**2 + y**2 <= max_distance**2) & (x**2 + y**2 >= min_distance**2)
    kernel = kernel.astype(float)
    
    total_tree_cover = scipy.signal.fftconvolve(tree_percent, kernel, mode='same')
    percent_trees = (total_tree_cover / kernel.sum()) * 100
    
    # Mask out trees and adjacent pixels
    percent_trees[np.where(adjacent_mask)] = np.nan
    percent_trees[percent_trees < 1] = 0

    da_percent_trees = xr.DataArray(
        percent_trees, 
        dims=("y", "x"),  
        coords={"y": tree_percent.coords["y"], "x": tree_percent.coords["x"]}, 
        name="percent_trees" 
    )
    
    return da_percent_trees


def shelter_categories(category_tif, height_tif=None, wind_ds=None, outdir='.', stub=None, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=True):
    """Define sheltered and unsheltered pixels
    
    Parameters
    ----------
        category_tif: Integer tif file generated by tree_categories.py
        height_tif: Integer tif file generated by apis.canopy_height.py
            - If not provided, then sheltered/unsheltered is defined by distance in pixels rather than tree heights
        wind_ds: NetCDF with eastward and westward wind speed generated by barra_daily.py
            - If not provided, then sheltered/unsheltered is defined by nearby tree density rather than distance from trees
        wind_method: Either 'MOST_COMMON', 'MAX', 'HAPPENED' or 'ANY'
            - MAX refers to the maximum wind speed
            - MOST_COMMON refers to the most common wind direction above the wind_threshold
            - HAPPENED refers to any direction where the winds exceeded the wind_threshold in wind_ds
            - ANY refers to all 8 compass directions
        wind_threshold: Integer in km/hr
        distance_threshold: The distance from trees that counts as sheltered.
            - Units are either 'tree heights' or 'number of pixels', depending on if a height_tif is provided
        density_threshold: The percentage tree cover within the distance_threshold that counts as sheltered
            - Only applies if the wind_ds is not provided.
        minimum_height: Assume that all tree pixels are at least this tall. If a height_tif is not provided, then all trees get assigned this height.
            
    Returns
    -------
        ds: an xarray with a band 'shelter_categories', where the integers represent the categories defined in 'shelter_category_labels'.

    Downloads
    ---------
        shelter_categories.tif: A tif file of the 'shelter_categories' band in ds, with colours embedded.
        shelter_categories.png: A png file like the tif file, but with a legend as well.
    
    """
    da_categories = rxr.open_rasterio(category_tif).squeeze('band').drop_vars('band')

    # I'm assuming that scattered trees don't count towards shelter for the purposes of blocking the wind, but do contribute to percentage tree cover
    tree_mask = da_categories >= 10 && da_categories < 20
    shelter = da_categories >= 12

    # Since the input tif is categorized, we assume here that any 'tree pixel' is 100% tree cover.
    tree_percent = tree_mask.astype(float)

    if height_tif:
        da_heights = rxr.open_rasterio(height_tif).squeeze('band').drop_vars('band')
        da_heights_reprojected = da_heights.rio.reproject_match(da_categories) 
        da_heights_nan = xr.where(shelter, da_heights_reprojected, np.nan)  # I think xr.where() is more readable than da.where()
        shelter_heights = xr.where(da_heights_nan <= minimum_height, minimum_height, da_heights_nan)
    else:
        shelter_heights = xr.where(shelter, minimum_height, np.nan)  

    if wind_ds:
        ds_wind = xr.load_dataset(wind_ds)
    
        if wind_method == 'MAX':
            df, max_speed, direction_max_speed = wind_dataframe(ds_wind)
            primary_wind_direction = direction_max_speed
            distances = compute_distance_to_tree_TH(shelter_heights, primary_wind_direction, distance_threshold)
            sheltered = distances > 0
        
        elif wind_method == 'MOST_COMMON':
            primary_wind_direction, _ = dominant_wind_direction(ds_wind, wind_threshold)
            distances = compute_distance_to_tree_TH(shelter_heights, primary_wind_direction, distance_threshold)
            sheltered = distances > 0
    
        elif wind_method == 'HAPPENED':
            _, df_wind = dominant_wind_direction(ds_wind, wind_threshold)
            strong_wind_directions = list(df_wind.loc[df_wind['Count'] > 0, 'Direction'])
            distance_rasters = []
            for wind_direction in strong_wind_directions:
                distances = compute_distance_to_tree_TH(shelter_heights, wind_direction, distance_threshold)
                distance_rasters.append(distances)
            masked_stack = xr.concat(masked_rasters, dim="stack")
            min_distances = masked_stack.min(dim="stack", skipna=True)
            sheltered = min_distances > 0
            
        elif wind_method == 'ANY':
            wind_directions = list(direction_map.keys())
            distance_rasters = []
            for wind_direction in wind_method:
                distances = compute_distance_to_tree_TH(shelter_heights, wind_direction, distance_threshold)
                distance_rasters.append(distances)
            masked_stack = xr.concat(masked_rasters, dim="stack")
            min_distances = masked_stack.min(dim="stack", skipna=True)
            sheltered = min_distances > 0
    else:
        da_percent_trees = computer_tree_densities(tree_percent)
        sheltered = da_percent_trees >= density_threshold

    # Assigning sheltered pixels to the label "31"
    da_shelter_categories = da_categories.where(~sheltered, 31)
    ds = da_shelter_categories.to_dataset(name='shelter_categories').drop_vars('spatial_ref')

    if not stub:
        # Use the same prefix as the original category_tif
        stub = category_tif.split('/')[-1].split('.')[0]

    if savetif:
        filename = os.path.join(outdir,f"{stub}_shelter_categories.tif")
        tif_categorical(ds['shelter_categories'], filename, shelter_categories_cmap)
    
    if plot:
        filename_png = os.path.join(outdir, f"{stub}_shelter_categories.png")
        visualise_categories(ds['shelter_categories'], filename_png, shelter_categories_cmap, shelter_categories_labels, "Shelter Categories")

    return ds


if __name__ == '__main__':

    # outdir = '../../../outdir/'
    # stub = 'TEST'
    # category_tif = "../../../outdir/TEST_categorised.tif"
    # height_tif = "../../../outdir/TEST_canopy_height.tif"
    # wind_ds = "../../../outdir/TEST_barra_daily.nc"
    # wind_method = 'MOST_COMMON'
    # wind_threshold = 15
    # distance_threshold = 20
    print()

outdir = '../../../outdir/'
stub = 'g2_26729'
category_tif = f"{outdir}{stub}_categorised.tif"
height_tif = f"{outdir}{stub}_canopy_height.tif"
wind_ds = f"{outdir}{stub}_barra_daily.nc"
wind_method = 'MOST_COMMON'
wind_threshold = 25
distance_threshold = 20
minimum_height = 1
wind_dir='E'
max_distance=20
density_threshold=10

da_categories = rxr.open_rasterio(category_tif).squeeze('band').drop_vars('band')
da_heights = rxr.open_rasterio(height_tif).squeeze('band').drop_vars('band')
shelter = da_categories >= 12
ds_wind = xr.load_dataset(wind_ds)

da_heights_reprojected = da_heights.rio.reproject_match(da_categories) 
da_heights_nan = xr.where(shelter, da_heights_reprojected, np.nan)  # I think xr.where() is more readable than da.where()
shelter_heights = xr.where(da_heights_nan <= minimum_height, minimum_height, da_heights_nan)

primary_wind_direction, df_wind = dominant_wind_direction(ds_wind, wind_threshold)

tree_mask = (da_categories >= 10) & (da_categories < 20)
tree_percent = tree_mask.astype(float)

da_percent_trees = computer_tree_densities(tree_percent)
sheltered = da_percent_trees >= density_threshold

sheltered.plot()

list(direction_map.keys())
