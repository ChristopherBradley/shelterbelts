import os
import argparse

import numpy as np
import xarray as xr
import rioxarray as rxr
import scipy

from shelterbelts.apis.barra_daily import wind_dataframe, dominant_wind_direction
from shelterbelts.indices.tree_categories import tree_categories_labels, tree_categories_cmap
from shelterbelts.apis.worldcover import tif_categorical, visualise_categories


shelter_categories_cmap = {
    0:(0, 0, 0),
    2:(150,150,150)
}
shelter_categories_labels = {
    0:'Unsheltered',
    2:'Sheltered'
}
shelter_categories_labels = tree_categories_labels | shelter_categories_labels
shelter_categories_cmap = tree_categories_cmap | shelter_categories_cmap

inverted_labels = {v: k for k, v in shelter_categories_labels.items()}


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
    
    new_hits = xr.where(shelter_heights.isnull(), shifted_full, np.nan) 
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
        
        new_hits = xr.where(shelter_heights.isnull(), shifted_full_centre, np.nan) 
        distances_centre = (shifted_full_max_centre - new_hits)
        
        distances = distances.where(~distances.isnull(), distances_centre)
    
    return distances


def computer_tree_densities(tree_percent, min_distance=0, max_distance=20, mask_adjacencies=False):
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
    
    # Mask out trees
    percent_trees[np.where(tree_mask)] = np.nan
    percent_trees[percent_trees < 1] = 0

    if mask_adjacencies:
        # We probably don't want to mask adjacencies for the first paper estimating amount of sheltered farmland in Australia
        # We probably do want to mask adjacencies for the second paper analysing the effects of tree cover on nearby productivity
        percent_trees[np.where(adjacent_mask)] = np.nan

    da_percent_trees = xr.DataArray(
        percent_trees, 
        dims=("y", "x"),  
        coords={"y": tree_percent.coords["y"], "x": tree_percent.coords["x"]}, 
        name="percent_trees" 
    )
    
    return da_percent_trees


def shelter_categories(category_tif, wind_ds=None, height_tif=None, outdir='.', stub='TEST', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, ds=None, savetif=True, plot=True):
    """Define sheltered and unsheltered pixels
    
    Parameters
    ----------
        category_tif: Integer tif file generated by tree_categories.py
        wind_ds: NetCDF with eastward and westward wind speed generated by barra_daily.py
            - If not provided, then sheltered/unsheltered is defined by nearby tree density rather than distance from trees
        height_tif: Integer tif file generated by apis.canopy_height.py
            - If not provided, then sheltered/unsheltered is defined by distance in pixels rather than tree heights
        outdir: The output directory to save the results.
        stub: Prefix for output files.
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
        minimum_height: Assume that all tree pixels are at least this tall. 
            - Only applies if a the height_tif is provided. 
        ds: Used instead of the category_tif when provided to avoid writing and reading lots of data.
            
    Returns
    -------
        ds: an xarray with a band 'shelter_categories', where the integers represent the categories defined in 'shelter_category_labels'.

    Downloads
    ---------
        shelter_categories.tif: A tif file of the 'shelter_categories' band in ds, with colours embedded.
        shelter_categories.png: A png file like the tif file, but with a legend as well.
    
    """
    if not ds:
        da_categories = rxr.open_rasterio(category_tif).squeeze('band').drop_vars('band')
    else:
        da_categories = ds['tree_categories']

    # I'm assuming that scattered trees don't count towards shelter for the purposes of blocking the wind, but do contribute to percentage tree cover
    tree_mask = (da_categories >= 10) & (da_categories < 20)
    shelter = da_categories >= 12

    # Since the input tif is categorized, we assume here that any 'tree pixel' is 100% tree cover.
    tree_percent = tree_mask.astype(float)

    if height_tif:
        da_heights = rxr.open_rasterio(height_tif).squeeze('band').drop_vars('band')
        da_heights_reprojected = da_heights.rio.reproject_match(da_categories) 

        # Using da.where instead of xr.where to preserve the xr.rio.crs
        da_heights_nan = da_heights_reprojected.where(~shelter, np.nan)  
        shelter_heights = da_heights_nan.where(da_heights_nan <= minimum_height, minimum_height)

        pixel_size = 10 # metres
        shelter_heights = (shelter_heights / pixel_size) * distance_threshold  # Scale the tree heights by the distance threshold
    else:
        # shelter_heights = xr.where(shelter, distance_threshold, np.nan)  
        shelter_heights = shelter.where(shelter, other=np.nan) * distance_threshold # preserving the rio.crs

    if wind_ds:
        ds_wind = xr.load_dataset(wind_ds)
    
        if wind_method == 'MAX':
            _, _, direction_max_speed = wind_dataframe(ds_wind)
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
            if not distance_rasters:
                sheltered = xr.full_like(shelter_heights, False).astype(bool)
            else:
                masked_stack = xr.concat(distance_rasters, dim="stack")
                min_distances = masked_stack.min(dim="stack", skipna=True)
                sheltered = min_distances > 0
                
        elif wind_method == 'ANY':
            wind_directions = list(direction_map.keys())
            distance_rasters = []
            for wind_direction in wind_directions:
                distances = compute_distance_to_tree_TH(shelter_heights, wind_direction, distance_threshold)
                distance_rasters.append(distances)
            masked_stack = xr.concat(distance_rasters, dim="stack")
            min_distances = masked_stack.min(dim="stack", skipna=True)
            sheltered = min_distances > 0
    else:
        da_percent_trees = computer_tree_densities(tree_percent)
        sheltered = da_percent_trees >= density_threshold

    # Assigning sheltered pixels a new label
    da_shelter_categories = da_categories.where(~sheltered, inverted_labels['Sheltered'])
    ds = da_shelter_categories.to_dataset(name='shelter_categories')

    if savetif:
        filename = os.path.join(outdir,f"{stub}_shelter_categories.tif")
        tif_categorical(ds['shelter_categories'], filename, shelter_categories_cmap)
    
    if plot:
        filename_png = os.path.join(outdir, f"{stub}_shelter_categories.png")
        visualise_categories(ds['shelter_categories'], filename_png, shelter_categories_cmap, shelter_categories_labels, "Shelter Categories")

    # ds = ds.rename({'x':'longitude', 'y': 'latitude'})

    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', help='Integer tif file generated by tree_categories.py')
    parser.add_argument('--wind_ds', help='NetCDF with eastward and westward wind speed generated by barra_daily.py')
    parser.add_argument('--height_tif', help='Integer tif file generated by apis.canopy_height.py')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', default='TEST', help='Prefix for output files.')
    parser.add_argument('--wind_method', default='MOST_COMMON', help="Either 'MOST_COMMON', 'MAX', 'HAPPENED' or 'ANY'")
    parser.add_argument('--distance_threshold', default=20, help='The distance from trees that counts as sheltered.')
    parser.add_argument('--wind_threshold', default=15, help='The wind speed used to determine the dominant wind direction.')
    parser.add_argument('--minimum_height', default=10, help="Assume that all tree pixels are at least this tall.")
    parser.add_argument('--density_threshold', default=10, help="The minimum percentage tree cover that counts as sheltered.")
    parser.add_argument('--plot', default=False, action="store_true", help="Boolean to Save a png file along with the tif")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    filename = args.filename
    wind_ds = args.wind_ds
    height_tif = args.height_tif
    outdir = args.outdir
    stub = args.stub
    wind_method = args.wind_method
    distance_threshold = int(args.distance_threshold)
    wind_threshold = int(args.wind_threshold)
    minimum_height = int(args.minimum_height)
    density_threshold = int(args.density_threshold)
    plot = args.plot

    ds = shelter_categories(filename, wind_ds, height_tif, outdir, stub, wind_method, wind_threshold, distance_threshold, density_threshold, minimum_height, savetif=True, plot=plot)


# +
# outdir = '../../../outdir/'
# stub = 'g2_26729'
# category_tif = f"{outdir}{stub}_categorised.tif"
# height_tif = f"{outdir}{stub}_canopy_height.tif"
# wind_ds = f"{outdir}{stub}_barra_daily.nc"
# wind_method = 'MOST_COMMON'
# wind_threshold = 25
# distance_threshold = 20
# minimum_height = 1
# wind_dir='E'
# max_distance=20
# density_threshold=10
