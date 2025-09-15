

import os
import rioxarray as rxr
import numpy as np
import xarray as xr
from skimage.morphology import skeletonize, disk
from scipy.ndimage import distance_transform_edt
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion


outdir = '../../../outdir'

filename_linear = os.path.join(outdir, 'shelter_indices_linear_categories.tif')
filename_labelled = os.path.join(outdir, 'shelter_indices_labelled_categories.tif')
filename_patch_matrics = os.path.join(outdir, 'shelter_indices_patch_metrics.csv')

da_linear = rxr.open_rasterio(filename_linear).isel(band=0).drop_vars('band')

da_labelled = rxr.open_rasterio(filename_labelled).isel(band=0).drop_vars('band')

df_patch_metrics = pd.read_csv(filename_patch_matrics)

df_patch_metrics

da_labelled.plot()


# +
def width_stats_per_group(da_labelled):
    results = []

    labels = np.unique(da_labelled.values)
    labels = labels[labels != 0]  # 0 is background

    for lbl in labels:
        mask = (da_labelled.values == lbl)

        stats = {"label": int(lbl)}

        disk_size = 2  # There doesn't seem to be much difference in the resulting average widths regardless of the disk size. 2 seems like a reasonable compromise.
        selem = disk(disk_size) 
        mask = binary_dilation(mask, structure=selem)
        mask = binary_erosion(mask, structure=selem)

        skel = skeletonize(mask)
        
        # Distance to the nearest background pixel
        dist = distance_transform_edt(mask)
        widths = dist[skel] * 2  # diameter = radius x 2

        if len(widths) > 0:
            stats = stats | {
                f"disk{disk_size}_mean_width": float(np.mean(widths)),  # mean is probably more intuitive than median
            }
        results.append(stats)

    return pd.DataFrame(results)

df_widths = width_stats_per_group(da_labelled)
df_widths

# +
import numpy as np
from skimage.morphology import skeletonize

def skeletonize_labels(da_labelled):
    # Start with empty raster
    skel_arr = np.zeros_like(da_labelled.values, dtype=np.int32)
    cleaned_arr = np.zeros_like(da_labelled.values, dtype=np.int32)

    labels = np.unique(da_labelled.values)
    labels = labels[labels != 0]  # skip background if 0 is background

    for lbl in labels:
        mask = (da_labelled.values == lbl)

        selem = disk(2)  # structuring element, adjust size
        mask = binary_dilation(mask, structure=selem)
        mask = binary_erosion(mask, structure=selem)
        
        # skeletonize the binary mask
        skel = skeletonize(mask)

        # burn label into skeleton raster
        skel_arr[skel] = lbl
        cleaned_arr[mask] = lbl

    # Create new DataArray with same coords/attrs as input
    da_skel = da_labelled.copy(data=skel_arr)
    da_cleaned = da_labelled.copy(data=cleaned_arr)

    return da_skel, da_cleaned

# Example usage
da_skeletons, da_cleaned = skeletonize_labels(da_labelled)

# Quick visualization
da_skeletons.plot(cmap="tab20")

# Export to GeoTIFF
da_skeletons.rio.to_raster("tree_skeletons.tif")

# -


