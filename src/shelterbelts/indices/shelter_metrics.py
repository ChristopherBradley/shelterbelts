# !pip install xlsxwriter

import os
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from scipy import ndimage
from scipy.stats import mode


from shelterbelts.indices.buffer_categories import buffer_categories_labels, buffer_categories_cmap
from shelterbelts.indices.tree_categories import tree_clusters

# +
# Full list of categories for reference:
# 0:'Not trees'
# 11:'Scattered Trees',
# 12:'Patch Core',
# 13:'Patch Edge',
# 14:'Corridor (other)',
# 15:'Trees in gullies',
# 16:'Trees on ridges',
# 17:'Trees next to roads' 
# 31: "Unsheltered Grassland",
# 32: "Sheltered Grassland",
# 41: "Unsheltered Cropland",
# 42: "Sheltered Cropland",
# 50: 'Built-up',
# 60: 'Bare / sparse vegetation',
# 70: 'Snow and ice',
# 80: 'Permanent water bodies',
# -

linear_cmap = {
    18:  (168, 131, 50), # The old corridor colour for reference (22, 212, 0)
    19: (91, 153, 75)  # Similar to the patch edges because similar function
}
linear_labels = {
    18: "Linear Patches",
    19: "Non-linear Patches"
}

linear_categories_cmap = buffer_categories_cmap | linear_cmap
linear_categories_labels = buffer_categories_labels | linear_labels

# Mapping for broader categories
landcover_groups = {
    10: "Trees",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    80: "Water"
}
def group_label(cat_id):
    """Apply broader group categories to each pixel"""
    return landcover_groups.get((cat_id // 10) * 10, 'Other')  # Flooring to the nearest 10


# Need to make sure you've pip installed xlsxwriter for this to work
def save_excel_sheets(dfs, filename):
    """Save each dataframe in a separate sheet in an excel file, and resize columns nicely
    dfs should be a dictionary of dataframes."""
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
    
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
    
            # Set the column widths to the longest item in that column
            for i, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                )
                worksheet.set_column(i + 1, i + 1, max_len)
                
            # Also set the index column width
            max_idx_len = max(df.index.astype(str).map(len).max(), len(str(df.index.name or "")))
            worksheet.set_column(0, 0, max_idx_len)
    
    print("Saved:", filename)


def class_metrics(buffer_tif, outdir=".", stub="TEST", save_excel=True):
    """Calculate the percentage cover in each class.
    
    Parameters
    ----------
        buffer_tif: A tif file where the integers represent the categories defined in 'buffer_category_labels'.

    Returns
    -------
        dfs: A dictionary with 4 dataframes: overall, landcover, trees, shelter

    Downloads
    ---------
        class_metrics.xlsx: An excel file with each dataframe in a separate tab.

    """
    da = rxr.open_rasterio(buffer_tif).isel(band=0)

    # Overall stats per category
    counts = da.values.ravel()
    counts = pd.Series(counts).value_counts().sort_index()
    total_pixels = da.shape[0] * da.shape[1]
    df_overall = pd.DataFrame({
        'category_id': counts.index,
        'label': [buffer_categories_labels.get(cat, 'Unknown') for cat in counts.index],
        'pixel_count': counts.values,
        'percentage': (counts.values / total_pixels) * 100
    })
    df_overall = df_overall.set_index('label')
    
    # Landcover groups
    df_overall['landcover_group'] = df_overall['category_id'].apply(group_label)
    df_landcover = df_overall.groupby('landcover_group')[['pixel_count', 'percentage']].sum()
    df_landcover = df_landcover.sort_values(by='percentage', ascending=False)
    df_landcover['percentage'] = df_landcover['percentage'].round(2)
    
    # Tree groups
    df_trees = df_overall[df_overall['landcover_group'] == 'Trees'].copy()
    total_tree_pixels = df_trees['pixel_count'].sum()
    df_trees['percentage'] = (df_trees['pixel_count'] / total_tree_pixels) * 100
    df_trees['percentage'] = df_trees['percentage'].round(2)
    df_trees = df_trees.drop(columns=['category_id', 'landcover_group'])
    df_trees = df_trees.sort_values(by='percentage', ascending=False)

    # Shelter groups
    df_production = df_overall[df_overall.index.str.contains("sheltered", case=False)].copy()
    df_production['shelter_status'] = df_production.index.str.split().str[0] # First word is the shelter status
    df_production['production_category'] = df_production.index.str.split().str[1]  # Last word is Grassland or Cropland
    df_summary = df_production.pivot_table(
        index='production_category',
        columns='shelter_status',
        values='pixel_count',
        aggfunc='sum',
        fill_value=0
    )
    df_summary.loc['Total'] = df_summary.sum()
    df_shelter = df_summary.div(df_summary.sum(axis=1), axis=0) * 100
    df_shelter = df_shelter.round(2)

    dfs = {
        "Overall": df_overall,
        "Landcover": df_landcover,
        "Trees": df_trees,
        "Shelter": df_shelter,
    }

    if save_excel:
        filename = os.path.join(outdir, f"{stub}_class_metrics.xlsx")
        save_excel_sheets(dfs, filename)

    return dfs


def patch_metrics(geometry, folder):
    """Calculate the length, width, height, direction, area, perimeter for each patch.
        Also calculates an overall mean and standard deviation for each attribute across all patches.
    
    Parameters
    ----------
        buffer_tif: A tif file where the integers represent the categories defined in 'buffer_category_labels'.

    Returns
    -------
        ds: xarray.DataSet with cleaned_categories and labelled_categories
        df: Individual attributes for each cluster.

    Downloads
    ---------
        patch_metrics.csv: A csv file with stats for each cluster.
        cleaned_categories.tif (dtype uint8): The buffer categories after applying a majority filter, pixel-wise and cluster-wise.
        cleaned_categories.png: Same as cleaned_categories.tif, but including a legend.
        labelled_categories.tif: A tif file with a unique identifier for each tree cluster. Note this does not contain the scattered trees. 
            - Uses dtype uint16, since there can theoretically be more than 256 patches.
        labelled_categories.png: A png for visualising the cluster id's and an ellipse around each cluster.

    """


# Inputs
outdir = "../../../outdir/"
stub = "shelter_indices"
buffer_tif = os.path.join(outdir, f"{stub}_buffer_categories.tif")

# Class metrics
dfs = class_metrics(buffer_tif, outdir, stub)

# Patch metrics
from shelterbelts.indices.buffer_categories import buffer_categories, buffer_categories_cmap, buffer_categories_labels
from shelterbelts.apis.worldcover import visualise_categories

da = rxr.open_rasterio(buffer_tif).isel(band=0)
visualise_categories(da, None, buffer_categories_cmap, buffer_categories_labels, "Buffer Categories")


def majority_filter_override(data, footprint, classes_that_override, classes_to_override):
    """apply majority filter to override specific class values"""
    data = data.copy()

    override_set = set(classes_that_override)
    target_set = set(classes_to_override)

    def selective_majority(values, center_val):
        # Count frequency of all values in the override set
        override_vals = [v for v in values if v in override_set]
        if not override_vals:
            return center_val  # keep original

        counts = np.bincount(override_vals)
        most_common = counts.argmax()
        max_count = counts[most_common]

        # Majority if it appears more than half of override_vals
        if max_count > len(override_vals) // 2:
            return most_common
        else:
            return center_val  # keep original

    # Pad the array so center pixel is accessible in each window
    padded = np.pad(data, pad_width=footprint.shape[0] // 2, mode='edge')
    result = data.copy()

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] in target_set:
                # Extract neighborhood
                y0 = i
                x0 = j
                win = padded[y0:y0 + footprint.shape[0], x0:x0 + footprint.shape[1]]
                values = win[footprint]
                new_val = selective_majority(values, data[i, j])
                result[i, j] = new_val

    return result



# +
# Pixel-wise majority filter to cleanup straggler pixels
radius = 3
y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
# disk = (x**2 + y**2) <= radius**2  # circular footprint
disk = np.ones((2*radius+1, 2*radius+1), dtype=bool) # Square footprint

filtered_array = da.data 
num_passes = 1
filtered_array = majority_filter_override(da.data, disk, [14, 15, 16, 17], [14, 15, 16, 17])  # Cleanup buffer classes
filtered_array = majority_filter_override(filtered_array, disk, [13], [14])  # Cleanup patch edges
da_filtered = xr.DataArray(filtered_array, coords=da.coords, dims=da.dims, attrs=da.attrs)

visualise_categories(da_filtered, None, buffer_categories_cmap, buffer_categories_labels, "Buffer Categories")

# +
# Assign cluster ids to the core areas and corresponding edges
arr = da_filtered.data  # shape: (200, 200)
core_mask = (arr == 12)
core_labels, num_labels = ndimage.label(core_mask)
distance, (inds_y, inds_x) = ndimage.distance_transform_edt(
    ~core_mask, return_indices=True
)
nearest_core_labels = core_labels[inds_y, inds_x]
assigned_labels = np.zeros_like(arr, dtype=np.int32)
assigned_labels[arr == 13] = nearest_core_labels[arr == 13]
assigned_labels[arr == 12] = core_labels[arr == 12]

# Assign ids to the rest of the patch types
buffer_category_ids = [14, 15, 16, 17]
for category_id in buffer_category_ids:
    da_category = (da_filtered == category_id)
    labelled_category = tree_clusters(da_category, max_gap_size=1)
    labelled_category = labelled_category + assigned_labels.max()
    labelled_arr = labelled_category.data
    mask = (arr == category_id)
    assigned_labels[mask] = labelled_arr[mask]

# Cluster-wise majority filter to reassign categories that have too few pixels
labels, counts = np.unique(assigned_labels, return_counts=True)
min_patch_size = 20
small_labels = labels[counts < min_patch_size]
small_mask = np.isin(assigned_labels, small_labels)
valid_mask = (~small_mask) & (assigned_labels != 0)
distance, (inds_y, inds_x) = ndimage.distance_transform_edt(
    ~valid_mask,
    return_indices=True
)
nearest_labels = assigned_labels[inds_y, inds_x]
assigned_labels[small_mask] = nearest_labels[small_mask]

# da_labelled = xr.DataArray(assigned_labels, coords=da.coords, dims=da.dims, attrs=da.attrs)
# da_labelled.rio.to_raster('TEST_labelled.tif')
# -

import matplotlib.pyplot as plt
plt.imshow(assigned_labels)
plt.colorbar()

from skimage.measure import regionprops
from matplotlib.patches import Ellipse

# Re-label the patches so they are consecutive integers
unique_labels = np.unique(assigned_labels)
unique_labels = unique_labels[unique_labels != 0]
new_labels = np.arange(1, len(unique_labels) + 1)
label_map = dict(zip(unique_labels, new_labels))
assigned_labels_relabelled = np.zeros_like(assigned_labels)
for old, new in label_map.items():
    assigned_labels_relabelled[assigned_labels == old] = new
assigned_labels = assigned_labels_relabelled

# Fit an ellipse around each category
props = regionprops(assigned_labels)

# +
# Create the patch metrics
results = []
for region in props:
    label_id = assigned_labels[region.coords[0][0], region.coords[0][1]]
    
    results.append({
        'label': label_id,
        'length': region.major_axis_length,
        'width': region.minor_axis_length,
        'perimeter': region.perimeter,
        'area': region.area,
        'orientation_degrees': np.degrees(region.orientation)
    })
df_patch_metrics = pd.DataFrame(results)

# I can add the class stats per patch, e.g. number of pixels in each category (or at least a category type, assuming I override the small categories)
# I can also assign an average height later using the canopy height from earlier
# could also add a better average width using skeletonization like in Aksoy 2009
# could also add the WSI and/or SNFI like in Liknes 2017, although probably allow any direction rather than just north/south and east/west

# -

df_patch_metrics.to_csv('patch_metrics.csv')

# Plot the clusters with ellipses around each one
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(assigned_labels, cmap='nipy_spectral', interpolation='nearest')
for region in props:
    y0, x0 = region.centroid  
    orientation = region.orientation
    length = region.major_axis_length
    width = region.minor_axis_length
    angle_deg = -np.degrees(orientation)
    ellipse = Ellipse(
        (x0, y0),             
        width=width,           
        height=length,       
        angle=angle_deg,         
        edgecolor='red',
        facecolor='none',
        linewidth=1
    )
    ax.add_patch(ellipse)

    ax.text(
        x0, y0,
        str(region.label),
        color='white',
        fontsize=12,
        ha='center',
        va='center'
    )
plt.title('Labelled Clusters')
plt.savefig('labelled_clusters.png')


# +
# Determine the most common category in each cluster
dominant_categories = []
for region in props:
    coords = region.coords
    cat_values = da_filtered.data[coords[:, 0], coords[:, 1]]
    most_common = mode(cat_values, axis=None, keepdims=False).mode
    dominant_categories.append(most_common)
    
df_patch_metrics["category_id"] = dominant_categories


# +
# Reclassify patches with core or edge dominant to just be a patch with core. 
# Might be interested in how much of the patch is each of these categories later
df_patch_metrics["category_name"] = df_patch_metrics["category_id"].map(buffer_categories_labels)
df_patch_metrics.loc[(df_patch_metrics['category_id'] == 12) | (df_patch_metrics['category_id'] == 13), 'category_name'] = 'Patch with core'

# Reclassifying patches with dominant core, to edge, to simplify reclassifications of corridor pixels to edges
df_patch_metrics.loc[(df_patch_metrics['category_id'] == 12) | (df_patch_metrics['category_id'] == 13), 'category_id'] = 13


# +
# Use the length/width ratio with a threshold of 2 to reassign corridor clusters to linear or non-linear
da_linear = da_filtered.copy()
df_patch_metrics['len/width'] = df_patch_metrics['length']/df_patch_metrics['width']

for i, row in df_patch_metrics.iterrows():
    if row["dominant_category"] == 14:
        label_id = row["label"]
        len_width_ratio = row["len/width"]

        # Ideally I'd use a more robust index like WSI, SNFI or some combination of a few like that
        new_class = 18 if len_width_ratio > 2 else 19  # 18 is linear features, 19 is non-linear features

        mask = (assigned_labels == label_id)
        da_linear.data[mask] = new_class
        df_patch_metrics.loc[i, 'category_id'] = new_class
        df_patch_metrics.loc[i, 'category_name'] = linear_labels[new_class]


# +
# Reassign the remaining corridor/other pixels to the corresponding cluster category (we could have done this before reclassifying corridor patches to linear/non-linear but it shouldn't matter)
remaining_mask = (da_linear.data == 14)
label_ids = assigned_labels[remaining_mask]

# Map each pixel's label to its new category
label_to_category = dict(zip(df_patch_metrics['label'], df_patch_metrics['category_id']))
mapped_categories = np.vectorize(label_to_category.get)(label_ids)
da_linear.data[remaining_mask] = mapped_categories

# -

visualise_categories(da_linear, None, linear_categories_cmap, linear_categories_labels, "Linear Categories")

# +
# Put all this code into a function
