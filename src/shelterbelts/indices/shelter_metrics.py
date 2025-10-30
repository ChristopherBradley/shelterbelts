# !pip install xlsxwriter

import os

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from scipy import ndimage
from scipy.stats import mode
from skimage.morphology import skeletonize, disk
from skimage.measure import regionprops
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

from shelterbelts.indices.buffer_categories import buffer_categories_labels, buffer_categories_cmap
from shelterbelts.indices.tree_categories import tree_clusters

from shelterbelts.apis.worldcover import visualise_categories, tif_categorical


linear_cmap = {
    18:  (168, 131, 50),  
    19: (91, 153, 75)  
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


def plot_clusters(assigned_labels, filename=None):
    """Visualise the clusters with ellipses and text ids"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(assigned_labels, cmap='nipy_spectral', interpolation='nearest')
    props = regionprops(assigned_labels)
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
    plt.title("Patch ID's")
    if filename:
        plt.savefig(filename)
        plt.close()
        print("Saved:", filename)
    else:
        plt.show()


def skeleton_stats(assigned_labels):
    """Use a skeleton after cleaning with binary dilation/erosion to calculate mean width and length"""
    results = []

    labels = np.unique(assigned_labels)
    labels = labels[labels != 0]  # 0 is background

    for lbl in labels:
        mask = (assigned_labels == lbl)

        stats = {"label": int(lbl)}

        disk_size = 2  # There doesn't seem to be much difference in the resulting average widths regardless of the disk size. 2 seems like a reasonable compromise.
        selem = disk(disk_size) 
        mask = binary_dilation(mask, structure=selem)
        mask = binary_erosion(mask, structure=selem)

        skel = skeletonize(mask)
        
        # Distance to the nearest background pixel
        dist = ndimage.distance_transform_edt(mask)
        widths = dist[skel] * 2  # diameter = radius x 2

        if len(widths) > 0:
            skeleton_width = float(np.mean(widths))
            skeleton_area = skel.sum()
            stats = stats | {
                f"skeleton_width": skeleton_width,  # mean is probably more intuitive than median
                f"skeleton_length": skeleton_area, 
                f"skeleton len/width": skeleton_area/skeleton_width # Might also be interesting to look at ellipse length / skeleton width, and/or incorporate the perimeter area ratio
            }
        results.append(stats)

    return pd.DataFrame(results)


def patch_metrics(buffer_tif, outdir=".", stub="TEST", ds=None, plot=True, save_csv=True, save_tif=True, save_labels=True, min_patch_size=20, crop_pixels=None):
    """Calculate patch metrics and cleanup the tree pixel categories.
    
    Parameters
    ----------
        buffer_tif: A tif file where the integers represent the categories defined in 'buffer_category_labels'.

    Returns
    -------
        ds: xarray.DataSet with linear_categories and labelled_categories
        df: Individual attributes for each cluster.

    Downloads
    ---------
        patch_metrics.csv: A csv file with stats for each cluster.
        linear_categories.tif (dtype uint8): The buffer categories after applying a majority filter, pixel-wise and cluster-wise.
        linear_categories.png: Same as linear_categories.tif, but including a legend.
        labelled_categories.tif: A tif file with a unique identifier for each tree cluster. 
            - No colour scheme applied, so can't view in Preview. Recommend viewing in QGIS with Paletted/Unique Values.
            - Doesn't contain the scattered trees. 
            - Uses dtype int32, since there can theoretically be more than 256 patches.
        labelled_categories.png: A png for visualising the cluster id's and an ellipse around each cluster.

    """
    # print(f"Starting patch_metrics for stub: {stub}")
    if not ds:
        da = rxr.open_rasterio(buffer_tif).isel(band=0)
    else:
        da = ds['buffer_categories']
    
    # Pixel-wise majority filter to cleanup straggler pixels
    radius = 3
    majority_disk = np.ones((2*radius+1, 2*radius+1), dtype=bool) # Square footprint seems to be better at fixing up diagonal stragglers
    filtered_array = majority_filter_override(da.data, majority_disk, [14, 15, 16, 17], [14, 15, 16, 17])  # Cleanup buffer classes
    filtered_array = majority_filter_override(filtered_array, majority_disk, [13], [14])  # Cleanup patch edges
    da_filtered = xr.DataArray(filtered_array, coords=da.coords, dims=da.dims, attrs=da.attrs)
    
    # Assign cluster ids to the core areas and corresponding edges
    arr = da_filtered.data  
    core_mask = (arr == 12)
    core_labels, num_labels = ndimage.label(core_mask)
    distance, (inds_y, inds_x) = ndimage.distance_transform_edt(
        ~core_mask, return_indices=True
    )
    nearest_core_labels = core_labels[inds_y, inds_x]
    assigned_labels = np.zeros_like(arr, dtype=np.int32)
    assigned_labels[arr == 13] = nearest_core_labels[arr == 13]
    assigned_labels[arr == 12] = core_labels[arr == 12]
    
    # Assign cluster ids for all the buffer groups (including small ones), since we're no longer converting these buffer categories.
    buffer_codes = [15, 16, 17]
    for buffer_code in buffer_codes:
        mask = (arr == buffer_code)
        labels, num_labels = ndimage.label(mask)
        assigned_labels[arr == buffer_code] = labels[arr == buffer_code]
    
    # Assign ids to the rest of the patch types
    # buffer_category_ids = [14, 15, 16, 17]  # Clean up all of the buffer categories. This runs into an issue if there aren't any large enough clusters nearby.
    buffer_category_ids = [14]  # Only clean up the "Other" category
    for category_id in buffer_category_ids:
        da_category = (da_filtered == category_id)
        labelled_category = tree_clusters(da_category, max_gap_size=1)
        labelled_category = labelled_category + assigned_labels.max()
        labelled_arr = labelled_category.data
        mask = (arr == category_id)
        assigned_labels[mask] = labelled_arr[mask]
    
    # Cluster-wise majority filter to reassign categories that have too few pixels
    labels, counts = np.unique(assigned_labels, return_counts=True)
    small_labels = labels[counts < min_patch_size]
    small_mask = np.isin(assigned_labels, small_labels)
    # valid_mask = (~small_mask) & (assigned_labels != 0)
    valid_mask = ((~small_mask) & (assigned_labels != 0)) | np.isin(arr, [15, 16, 17])  # Allow small buffer categories
    distance, (inds_y, inds_x) = ndimage.distance_transform_edt(
        ~valid_mask,
        return_indices=True
    )
    nearest_labels = assigned_labels[inds_y, inds_x]
    assigned_labels[small_mask] = nearest_labels[small_mask]
        
    # Re-label the patches so they are consecutive integers
    unique_labels = np.unique(assigned_labels)
    unique_labels = unique_labels[unique_labels != 0]
    new_labels = np.arange(1, len(unique_labels) + 1)
    label_map = dict(zip(unique_labels, new_labels))
    assigned_labels_relabelled = np.zeros_like(assigned_labels)
    for old, new in label_map.items():
        assigned_labels_relabelled[assigned_labels == old] = new
    assigned_labels = assigned_labels_relabelled

    # Find the skeleton of each cluster
    df_widths = skeleton_stats(assigned_labels) 

    # Fit an ellipse around each category 
    props = regionprops(assigned_labels)
    
    if plot:
        filename = os.path.join(outdir, f'{stub}_labelled_categories.png')
        plot_clusters(assigned_labels, filename)
    
    # Create the patch metrics
    results = []
    for region in props:
        label_id = assigned_labels[region.coords[0][0], region.coords[0][1]]
        
        results.append({
            'label': label_id,
            'ellipse_length': region.major_axis_length,
            'ellipse_width': region.minor_axis_length,
            'perimeter': region.perimeter,
            'area': region.area,
            'orientation_degrees': np.degrees(region.orientation)
        })
        # Later can add:
            # class stats per patch, e.g. number of pixels in each category (or at least a category type, assuming I override the small categories)
            # average height using the canopy height from earlier
            # better average width using skeletonization like in Aksoy 2009
            # Other indices like the WSI and/or SNFI from Liknes 2017, although adjust to allow any direction rather than just north/south and east/west

    df_patch_metrics = pd.DataFrame(results)
    
    # Determine the most common category in each cluster
    dominant_categories = []
    for region in props:
        coords = region.coords
        cat_values = da_filtered.data[coords[:, 0], coords[:, 1]]
        most_common = mode(cat_values, axis=None, keepdims=False).mode
        dominant_categories.append(most_common)
        
    df_patch_metrics["category_id"] = dominant_categories
    
    # Reclassify patches with core or edge dominant for consistency
    # Later, might be interested in the percentage of these two categories in each patch like the class metrics
    df_patch_metrics["category_name"] = df_patch_metrics["category_id"].map(linear_categories_labels)  # linear_labels
    df_patch_metrics.loc[(df_patch_metrics['category_id'] == 12) | (df_patch_metrics['category_id'] == 13), 'category_name'] = 'Patch with core'
    df_patch_metrics.loc[(df_patch_metrics['category_id'] == 12) | (df_patch_metrics['category_id'] == 13), 'category_id'] = 13
        
    # Use the length/width ratio to reassign corridor clusters to linear or non-linear
    da_linear = da_filtered.copy()
    if len(results) > 0:
        df_patch_metrics['ellipse len/width'] = df_patch_metrics['ellipse_length']/df_patch_metrics['ellipse_width']
        df_patch_metrics = df_patch_metrics.merge(df_widths)
    else:
        df_patch_metrics = df_widths
    
    # Remove any rows where the area is smaller than the min_patch_size. Necessary since I'm using small buffer groups to reassign "Other" classes.
    # import pdb; pdb.set_trace()  # Useful for debugging in a jupyter notebook
    
    if len(df_patch_metrics) > 0 and 'area' in df_patch_metrics.columns:
        df_patch_metrics_large = df_patch_metrics[df_patch_metrics['area'] > min_patch_size] 
    else:
        df_patch_metrics_large = df_patch_metrics  # Probably an empty list because there are no trees in the region
    
    if save_csv:
        filename = os.path.join(outdir, f'{stub}_patch_metrics.csv')
        df_patch_metrics_large.to_csv(filename, index=False)
        print("Saved:", filename)
    
    for i, row in df_patch_metrics.iterrows():
        if row["category_id"] == 14:
            label_id = row["label"]
            len_width_ratio = row["ellipse len/width"]
            
            if "skeleton len/width" not in row.index:
                new_class = 12  # I was trying to fix the random 1kmx1km tiles in ACT forests with all '19' values, but this line had no effect. Need to debug further.

            # Arbitrary thresholds that I need to play around with. Should add these as parameters to the function.
            elif row["ellipse len/width"] > 2 and row["skeleton len/width"] > 4:
                new_class = 18  # linear features
            else:
                new_class = 19  # non-linear features
    
            mask = (assigned_labels == label_id)
            da_linear.data[mask] = new_class
            df_patch_metrics.loc[i, 'category_id'] = new_class
            df_patch_metrics.loc[i, 'category_name'] = linear_categories_labels[new_class]


    # Reassign the remaining corridor/other pixels to the corresponding cluster's category 
    remaining_mask = (da_linear.data == 14)
    if (remaining_mask.sum() > 0):
        label_ids = assigned_labels[remaining_mask]

        if 'label' not in df_patch_metrics.columns:
            mapped_categories = [11] * len(label_ids) # Assuming the cluster has been cut off by water or another non-tree category, in which case we assign it to scattered_trees.
        else:
            label_to_category = dict(zip(df_patch_metrics['label'], df_patch_metrics['category_id']))  # There might be a case where df_patch_metrics['category_id'] is None? In which case I should also set to 11 for scattered trees.
            mapped_categories = np.vectorize(label_to_category.get)(label_ids)

        # import pdb; pdb.set_trace() # Useful for debugging in a jupyter notebook

        da_linear.data[remaining_mask] = mapped_categories

    # Maybe I should add an assert that there aren't any 14 labels left, since I'm not confident I've covered every scenario.

    # crop da_linear by n pixels on every side
    
    if plot:
        filename = os.path.join(outdir, f'{stub}_linear_categories.png')
        
        # I should also remove the really small patches from this plot and corresponding tif file for easier visualisation
        visualise_categories(da_linear, filename, linear_categories_cmap, linear_categories_labels, "Linear Categories")
    
    ds = da_linear.to_dataset(name="linear_categories")
    ds['labelled_categories'] = (["y", "x"], assigned_labels)

    if crop_pixels is not None:
        ds = ds.isel(
            x=slice(crop_pixels, -crop_pixels),
            y=slice(crop_pixels, -crop_pixels)
        )
    
    if save_tif:
        filename_linear = os.path.join(outdir, f'{stub}_linear_categories.tif')
        tif_categorical(ds['linear_categories'], filename_linear, linear_categories_cmap) 
    
        if save_labels:            
            filename_labelled = os.path.join(outdir, f'{stub}_labelled_categories.tif')
            ds['labelled_categories'].rio.to_raster(filename_labelled)  # Not applying a colour scheme because I prefer to use the QGIS 'Paletted/Unique' Values for viewing this raster
            print("Saved:", filename_labelled)

    return ds, df_patch_metrics


def class_metrics(buffer_tif, outdir=".", stub="TEST", ds=None, save_excel=True):
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
    if not ds:
        da = rxr.open_rasterio(buffer_tif).isel(band=0)
    else:
        da = ds['linear_categories']

    # Overall stats per category
    counts = da.values.ravel()
    counts = pd.Series(counts).value_counts().sort_index()
    total_pixels = da.shape[0] * da.shape[1]
    df_overall = pd.DataFrame({
        'category_id': counts.index,
        'label': [linear_categories_labels.get(cat, 'Unknown') for cat in counts.index],
        'pixel_count': counts.values,
        'percentage': (counts.values / total_pixels) * 100
    })
    df_overall = df_overall.set_index('label')
    
    # Landcover groups
    df_overall['landcover_group'] = df_overall['category_id'].apply(group_label) # Not sure why this was commented out?
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

import argparse
def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--buffer_tif', help='Integer tif file generated by buffer_categories.py')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', default=None, help='Prefix for output files.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    geotif = args.buffer_tif
    outdir = args.outdir
    stub = args.stub
    ds, df = patch_metrics(geotif, outdir, stub)

    geotif = os.path.join(outdir, f"{stub}_linear_categories.tif")
    dfs = class_metrics(geotif, outdir, stub)

# +
# patch_metrics("../../../outdir/hydrolines_buffer_categories.tif")
# -

# outdir = "../../../outdir/"
# stub = "shelter_indices"
# geotif = os.path.join(outdir, f"{stub}_buffer_categories.tif")
