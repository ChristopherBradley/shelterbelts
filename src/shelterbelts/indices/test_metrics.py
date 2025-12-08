from shelterbelts.indices.shelter_metrics import patch_metrics, pixel_majority_filter, assign_labels, split_disconnected_clusters, skeleton_stats


# +
# import os

# # # +
# import numpy as np
# import pandas as pd
# import rioxarray as rxr
# import xarray as xr
# from scipy import ndimage
# from scipy.stats import mode
# from skimage.morphology import skeletonize, disk
# from skimage.measure import regionprops, label
# from skimage.draw import ellipse_perimeter
# from skimage.graph import route_through_array

# from matplotlib.patches import Ellipse
# import matplotlib.pyplot as plt
# from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
# # -

# from shelterbelts.indices.buffer_categories import buffer_categories_labels, buffer_categories_cmap
# from shelterbelts.indices.tree_categories import tree_clusters
# from shelterbelts.indices.opportunities import segmentation

# from shelterbelts.apis.worldcover import visualise_categories, tif_categorical


# -

buffer_tif = '/scratch/xe2/cb8590/tmp/TEST2_buffer_categories.tif'
crop_pixels=20
min_patch_size=20

# %%time
# Exceeds memory allocation
ds, df = patch_metrics(buffer_tif, crop_pixels=20)



# +

# da = rxr.open_rasterio(buffer_tif).isel(band=0)


# # Pixel-wise majority filter to cleanup straggler pixels. Changes the da directly.
# da_filtered = pixel_majority_filter(da)

# # Crop the output if it was expanded before the pipeline started
# if crop_pixels is not None and crop_pixels != 0:
#     da_filtered = da_filtered.isel(
#         x=slice(crop_pixels, -crop_pixels),
#         y=slice(crop_pixels, -crop_pixels)
#     )

# # Assign labels and a cluster-wise majority filter on the labels (but doesn't change the da yet)
# assigned_labels = assign_labels(da_filtered, min_patch_size)
# assigned_labels = split_disconnected_clusters(assigned_labels)  # The ellipses go haywire if the clusters are not connected

# -

# # This function is causing the memory accumulation
# df_patch_metrics, ellipse_outline_raster, skeleton_raster, shortest_path_raster, perpendicular_raster, widths_raster = skeleton_stats(assigned_labels)



