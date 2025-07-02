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


from shelterbelts.apis.worldcover import tif_categorical

# Sample file
indir = "../../../data/"
outdir = "../../../outdir/"
filename = f'{indir}g2_26729_binary_tree_cover_10m.tiff'

# Load the sample data
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
ds = da.to_dataset(name='woody_veg').drop_vars('spatial_ref')
woody_veg = ds['woody_veg'].values.astype(bool)

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
edge_area = binary_dilation(core_area, structure=core_kernel) & ~core_area & woody_veg

plt.imshow(edge_area)

# Use the edges & core areas & scattered trees to define corridors
corridor_area = woody_veg & ~(core_area | edge_area | scattered_area)

plt.imshow(corridor_area)

# Double checking we don't assign different categories to the same pixel
combined = (
    scattered_area.astype(int) +
    core_area.astype(int) +
    edge_area.astype(int) +
    corridor_area.astype(int)
)
assert np.all(combined <= 1), "Category masks overlap — each pixel should belong to at most one category"
assert np.array_equal(combined > 0, woody_veg), "Combined areas don't match the original woody_veg"

# Create a single array with all the layers
tree_category_labels = {'scattered_area': 1, 'core_area':2, 'edge_area':3, 'corridor_area':4}
tree_categories_cmap = {
    0:(255, 255, 255),
    1:(122, 82, 0),
    2:(8, 79, 0),
    3:(14, 138, 0),
    4:(22, 212, 0)
}
tree_categories = np.zeros_like(woody_veg, dtype=np.uint8)
tree_categories[scattered_area] = tree_category_labels['scattered_area']
tree_categories[core_area]      = tree_category_labels['core_area']
tree_categories[edge_area]      = tree_category_labels['edge_area']
tree_categories[corridor_area]  = tree_category_labels['corridor_area']
ds['tree_categories'] = (('y', 'x'), tree_categories)

ds['tree_categories'].plot()

# Create a tif output with colours embedded
stub = filename.split('/')[-1].split('.')[0]
filename_categorical = f"{outdir}{stub}_categorised.tif"
tif_categorical(ds['tree_categories'], filename_categorical, tree_categories_cmap)

# +
# Create a visualisation in python
# -





def tree_categories(woody_veg_tif, min_patch_size=20, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True):
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
