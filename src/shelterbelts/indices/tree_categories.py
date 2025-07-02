# +
# Group trees together and calculate statistics including: 
# Number of shelterbelts w/ length, width, area, perimeter, height (min, mean, max) for each (and then mean and sd)
# Area of sheltered and unsheltered crop & pasture by region and different thresholds
# +
import os
import glob
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import pyproj
from scipy.ndimage import label, distance_transform_edt, gaussian_filter, binary_erosion, binary_dilation, generate_binary_structure

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# -


# Sample file
indir = "../../../data/"
filename = f'{indir}g2_26729_binary_tree_cover_10m.tiff'

# Load the sample data
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
ds = da.to_dataset(name='woody_veg').drop_vars('spatial_ref')
woody_veg = ds['woody_veg'].values



# +
# Assign a group label to trees within a given distance of each other
# Using a gap_size of 0 means only direct adjacencies with the 4 neighbour rule

# Create a circular kernel (as circular as you can get in a grid) with diameter = 1 + gap_size x 2 
max_gap_size = 2
y, x = np.ogrid[-max_gap_size:max_gap_size+1, -max_gap_size:max_gap_size+1]
gap_kernel = (x**2 + y**2 <= max_gap_size**2)

# Dilate the woody veg based on this kernel, to connect tree clusters that are close together
dilated = binary_dilation(woody_veg, structure=gap_kernel)

# Label the clusters
labeled_dilated, num_features = label(dilated)

# Remove the dilations
trees_labelled = labeled_dilated * woody_veg
# -

plt.imshow(trees_labelled)

plt.imshow(corridor_area)

# Create a raster of scattered_trees
min_patch_size = 20
counts = np.bincount(trees_labelled.ravel())
small_clusters = np.flatnonzero(counts < min_patch_size)
scattered_area = np.isin(trees_labelled, small_clusters)

plt.imshow(scattered_area)

# +
# Create a raster of core_areas
edge_size = 3
y, x = np.ogrid[-edge_size:edge_size+1, -edge_size:edge_size+1]
core_kernel = (x**2 + y**2 <= edge_size**2)
all_core_area = binary_erosion(woody_veg, structure=core_kernel)

# Exclude core area's smaller than the scattered trees threshold
labeled_cores, num_features = label(all_core_area)
large_cores = np.flatnonzero(np.bincount(labeled_cores.ravel()) >= min_patch_size)[1:] # Drop the 0 category
core_area = np.isin(labeled_cores, large_cores)
# -


plt.imshow(core_area)

# Find edges surrounding core areas
edge_factor = 2   # Extra distance to expand from the core area
double_edge_size = edge_size * edge_factor
y, x = np.ogrid[-double_edge_size:double_edge_size+1, -double_edge_size:double_edge_size+1]
edge_kernel = (x**2 + y**2 <= double_edge_size**2)
edge_area = binary_dilation(core_area, structure=edge_kernel * 2) & ~core_area & woody_veg

plt.imshow(edge_area)

# Use the edges & core areas & scattered trees to define corridors
corridor_area = woody_veg & ~(core_area | edge_area | scattered_area)

plt.imshow(corridor_area)









def tree_categories(woody_veg_tif, min_patch_size=20, edge_size=3, max_gap_size=2, ds=None, save_tif=True):
    """Categorise a boolean woody veg tif into scattered trees, edges, core areas, and corridors, based on the Fragstats landscape ecology approach

    Parameters
    ----------
        woody_veg_tif: A binary tif file containing tree/no tree information
        min_length: The minimum length (in any direction) to be classified as a patch/corrider rather than just scattered trees.
        edge_size: The buffer distance at the edge of a patch, with pixels inside this being the 'core area'. 
            Non-scattered tree pixels outside the core area but within edge_size pixels of a core area get defined as 'edge' pixels. Otherwise they become 'corridor' pixels.
        gap_size: The allowable gap between two tree clusters before considering them separate patches.
        ds: a pre-loaded xarray.DataSet with a band 'woody_veg'. This gets used instead of the woody_veg_tif when provided.
        save_tif: Boolean to determine whether to save the final result to file or not.

    Returns
    -------
        ds: an xarray with a band 'tree_categories', where the integers represent the categories defined in 'tree_category_labels'.

    Downloads
    ---------
        woody_veg_categorised.tif: A tif file of the 'tree_categories' band in ds, with colours embedded.
    
    """

if __name__ == '__main__':
    tree_categories("TEST.tif")
