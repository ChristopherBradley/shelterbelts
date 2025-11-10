import os

# +
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from scipy import ndimage
from scipy.stats import mode
from skimage.morphology import skeletonize, disk
from skimage.measure import regionprops, label
from skimage.draw import ellipse_perimeter
from skimage.graph import route_through_array

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
# -

from shelterbelts.indices.buffer_categories import buffer_categories_labels, buffer_categories_cmap
from shelterbelts.indices.tree_categories import tree_clusters
from shelterbelts.indices.opportunities import segmentation

from shelterbelts.apis.worldcover import visualise_categories, tif_categorical


linear_cmap = {
    18:  [190, 160, 60], # // Light brown: Linear patches (168, 131, 50),  
    19:  [165, 195, 45] # // Bright olive green: Non-linear patches (91, 153, 75)  
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


def pixel_majority_filter(da, radius=3):
    """Cleaning up straggler pixels by assigning to the majority in a given area"""
    majority_disk = np.ones((2*radius+1, 2*radius+1), dtype=bool) # Square footprint seems to be better at fixing up diagonal stragglers
    filtered_array = majority_filter_override(da.data, majority_disk, [14, 15, 16, 17], [14, 15, 16, 17])  # Cleanup buffer classes
    filtered_array = majority_filter_override(filtered_array, majority_disk, [13], [14])  # Cleanup patch edges
    da_filtered = xr.DataArray(filtered_array, coords=da.coords, dims=da.dims, attrs=da.attrs)
    return da_filtered


def assign_labels(da_filtered, min_patch_size=20):
    """Assign an id to each cluster of trees, and merge small clusters into nearby larger ones"""

    # This segmentation was too slow, and was segmenting core areas that I didn't want it to.
    # if segmentation:
    #     # Updating uncategorised labels if there are multiple segments
    #     mask = (da_filtered == 14).values
    #     skeletonized_uncategories = skeletonize(mask)
    #     segments = segmentation(skeletonized_uncategories)
    #     dist, nearest_idx = distance_transform_edt(segments == 0, return_indices=True)     # Compute distance transform to nearest segment pixel
    #     nearest_segments = segments[tuple(nearest_idx)]     # Get the segment ID for the nearest nonzero pixel
    #     offset = assigned_labels.max()     # Offset new segment IDs to avoid conflicts with existing ones
    #     new_labels = np.where(mask, nearest_segments + offset, assigned_labels)
    #     new_labels[(mask) & (nearest_segments == 0)] = 0     # Set background (segment==0) back to 0 in the new areas

    # Assign cluster ids to the core areas and corresponding edges
    arr = da_filtered.data  
    core_mask = (arr == 12)
    core_labels, num_labels = ndimage.label(core_mask)
    _, (inds_y, inds_x) = ndimage.distance_transform_edt(
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
    
    # Cluster-wise majority filter for very small clusters. Changes the labels. 
    # I do actually like these categorizations, but they get overwritten by the required resplitting before running skeleton stats right now. So I could probably remove this and just have simple labelling now.
    labels, counts = np.unique(assigned_labels, return_counts=True)
    small_labels = labels[counts < min_patch_size]
    small_mask = np.isin(assigned_labels, small_labels)
    # valid_mask = ((~small_mask) & (assigned_labels != 0)) | np.isin(arr, [15, 16, 17])  # Don't merge small buffer categories
    valid_mask = ((~small_mask) & (assigned_labels != 0))
    _, (inds_y, inds_x) = ndimage.distance_transform_edt(
        ~valid_mask,
        return_indices=True
    )
    nearest_labels = assigned_labels[inds_y, inds_x]
    assigned_labels[small_mask] = nearest_labels[small_mask]
        
    # Re-label the patches so they are consecutive integers
    _, labels_consecutive = np.unique(assigned_labels, return_inverse=True)
    assigned_labels = labels_consecutive.reshape(assigned_labels.shape)
    return assigned_labels


def split_disconnected_clusters(assigned_labels, connectivity=2):
    """Split disconnected parts of each label into separate labels."""
    new_labels = assigned_labels.copy()
    next_label = new_labels.max() + 1

    for lbl in np.unique(assigned_labels):
        if lbl == 0:
            continue

        mask = (assigned_labels == lbl)
        labeled_mask, n_components = ndimage.label(mask, structure=np.ones((3,3)) if connectivity==2 else None)

        # If more than one component, assign new labels
        for comp_id in range(1, n_components + 1):
            if comp_id == 1:
                # Keep the first component as original label
                continue
            new_labels[labeled_mask == comp_id] = next_label
            next_label += 1

    return new_labels


# +
def skeleton_stats(assigned_labels, min_patch_size=20, save_labels=True):
    """
    Minimal function to create skeleton raster and ellipse outline raster for debugging.
    
    Returns:
        skeleton_raster, ellipse_outline_raster
    """
    shortest_path_raster = np.zeros_like(assigned_labels, dtype=np.int32)
    ellipse_outline_raster = np.zeros_like(assigned_labels, dtype=np.int32)
    ellipse_endpoints_raster = np.zeros_like(assigned_labels, dtype=np.int32)
    widths_raster = np.zeros_like(assigned_labels, dtype=np.int32)
    perpendicular_raster = np.zeros_like(assigned_labels, dtype=np.int32)
    
    tree_mask = assigned_labels > 0
    
    # Precompute mask where a kernel fully fits outside trees
    edge_size = 3
    y, x = np.ogrid[-edge_size:edge_size + 1, -edge_size:edge_size + 1]
    kernel = (x**2 + y**2 <= (edge_size)**2)
    tree_counts = convolve(tree_mask.astype(np.uint8), kernel, mode='constant', cval=0)
    no_trees = (tree_counts == 0)
    safe_empty_mask = ndimage.convolve(no_trees.astype(np.uint8), kernel, mode='constant', cval=0) > 0
    safe_empty_mask = safe_empty_mask.astype(bool)
    dist_to_empty = ndimage.distance_transform_edt(~safe_empty_mask)

    results = []
    width_raster = np.zeros_like(tree_mask, dtype=float)
    props = regionprops(assigned_labels)
    for i, prop in enumerate(props):
        lbl = prop.label

        mask = (assigned_labels == lbl)

        if mask.sum() < min_patch_size:
            continue  # I might want to reindex these lbl's so they become consecutive again
            
        skel = skeletonize(mask)
        skeleton_raster[skel] = lbl

        # Ellipse parameters
        y0, x0 = prop.centroid
        orientation = prop.orientation
        a = prop.minor_axis_length / 2.0
        b = prop.major_axis_length / 2.0
        
        # Rasterize ellipse outline (skimage uses opposite orientation convention)
        rr, cc = ellipse_perimeter(
            int(round(y0)), int(round(x0)),
            int(round(b)), int(round(a)),
            orientation=-orientation,  
            shape=assigned_labels.shape
        )
        ellipse_outline_raster[rr, cc] = lbl

        # Calculate the endpoints
        angle = orientation
        x1 = x0 + np.sin(angle) * b
        y1 = y0 + np.cos(angle) * b
        x2 = x0 - np.sin(angle) * b
        y2 = y0 - np.cos(angle) * b

        # Find skeleton pixels closest to ellipse ends
        coords = np.column_stack(np.nonzero(skel))
        d1 = np.hypot(coords[:, 0] - y1, coords[:, 1] - x1)
        d2 = np.hypot(coords[:, 0] - y2, coords[:, 1] - x2)
        p1 = tuple(coords[np.argmin(d1)])
        p2 = tuple(coords[np.argmin(d2)])

        # Visualise the endpoints for debugging
        # ellipse_endpoints_raster[p1] = lbl
        # ellipse_endpoints_raster[p2] = lbl
        
        # Find the shortest path along the skeleton to each end of the ellipse
        cost = np.where(skel, 1, np.inf)
        try:
            path, _ = route_through_array(cost, p1, p2, fully_connected=True)
            skel_new = np.zeros_like(skel, dtype=bool)
            for r, c in path:
                skel_new[r, c] = True
            skel = skel_new
        except Exception:
            # If pathfinding fails, keep original skeleton
            pass
        
        shortest_path_raster[skel] = lbl

        # Calculate the widths
        ys, xs = np.nonzero(skel)
                
        # perpendicular directions (chatgpt always gets the sin and cos the wrong way around)
        dy = np.cos(orientation + np.pi / 2)
        dx = np.sin(orientation + np.pi / 2)
        
        # step 1 pixel perpendicular (round to nearest int)
        dy_int = np.round(dy).astype(int)
        dx_int = np.round(dx).astype(int)
        
        # coordinates of the two perpendicular neighbors
        y1 = np.clip(ys + dy_int, 0, dist_to_empty.shape[0] - 1)
        x1 = np.clip(xs + dx_int, 0, dist_to_empty.shape[1] - 1)
        y2 = np.clip(ys - dy_int, 0, dist_to_empty.shape[0] - 1)
        x2 = np.clip(xs - dx_int, 0, dist_to_empty.shape[1] - 1)

        perpendicular_raster[y1, x1] = lbl
        perpendicular_raster[y2, x2] = lbl
        
        # sum the distances from both sides
        v1 = dist_to_empty[y1, x1]
        v2 = dist_to_empty[y2, x2]
        
        # widths_raster[ys, xs] = v1 + v2 + 1 # The skeleton must be at least 1 wide
        widths = v1 + v2 + 1
        widths_raster[skel] = widths

        if len(widths) > 0:
            skeleton_width = float(np.mean(widths))
            skeleton_area = skel.sum()
            stats = {
                'label': lbl,
                'ellipse_length': prop.major_axis_length,
                'ellipse_width': prop.minor_axis_length,
                'ellipse len/width': prop.major_axis_length/prop.minor_axis_length,
                'perimeter': prop.perimeter,
                'area': prop.area,
                'orientation_degrees': np.degrees(prop.orientation),
                f"skeleton_width": skeleton_width,  # mean is probably more intuitive than median
                f"skeleton_length": skeleton_area, 
                f"skeleton len/width": skeleton_area/skeleton_width 
            }
            results.append(stats)

    df = pd.DataFrame(results)

    if save_labels:
        
    
    return df, ellipse_outline_raster, skeleton_raster, shortest_path_raster, perpendicular_raster, widths_raster


# -

def patch_metrics(buffer_tif, outdir=".", stub="TEST", ds=None, plot=True, save_csv=True, save_tif=True, save_labels=True, 
                  min_shelterbelt_length=20, max_shelterbelt_width=4, min_patch_size=20, crop_pixels=None):
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
    
    # Pixel-wise majority filter to cleanup straggler pixels. Changes the da directly.
    da_filtered = pixel_majority_filter(da)

    # Crop the output if it was expanded before the pipeline started
    if crop_pixels is not None and crop_pixels != 0:
        da_filtered = da_filtered.isel(
            x=slice(crop_pixels, -crop_pixels),
            y=slice(crop_pixels, -crop_pixels)
        )

    # Assign labels and a cluster-wise majority filter on the labels (but doesn't change the da yet)
    assigned_labels = assign_labels(da_filtered, min_patch_size)
    assigned_labels = split_disconnected_clusters(assigned_labels)  # The ellipses go haywire if the clusters are not connected
    
    # Fit an ellipse around each cluster 
    props = regionprops(assigned_labels)
    
    # Find the skeleton of each cluster
    df_patch_metrics, ellipse_outline_raster, skeleton_raster, shortest_path_raster, perpendicular_raster, widths_raster = skeleton_stats(assigned_labels2)

    # Save these intermediate rasters
    if save_tif and save_labels:
        ds_labels = da_filtered.to_dataset(name="filtered")
        raster_dict = {
            'assigned_labels':assigned_labels,
            'ellipse_outline_raster':ellipse_outline_raster,
            'shortest_path_raster':shortest_path_raster,
            'perpendicular_raster':perpendicular_raster,
            'widths_raster':widths_raster
        }
        for raster_key, raster_variable in raster_dict.items():
            ds_labels[raster_key] = (["y", "x"], raster_variable)
            filename_labelled = os.path.join(outdir, f'{stub}_{raster_key}.tif')
            ds_labels[raster_key].astype(float).rio.to_raster(filename_labelled)  # Not applying a colour scheme because I prefer to use the QGIS 'Paletted/Unique' values for these rasters
            print("Saved:", filename_labelled)

    # Determine the most common category for each row in the patch metrics
    dominant_categories = []
    for lbl in df['label']:
        lbl_categories = da_filtered.where(assigned_labels == lbl).values
        values, counts = np.unique(lbl_categories[~np.isnan(lbl_categories)], return_counts=True)
        most_common = values[np.argmax(counts)]
        dominant_categories.append(most_common)
    df_patch_metrics["category_id"] = dominant_categories

    # # Reassign edges so they get merged with their corresponding cores in the output csv. I think I prefer to keep them separate actually - more intuitive.
    # df_patch_metrics["category_name"] = df_patch_metrics["category_id"].map(linear_categories_labels)  # linear_labels
    # df_patch_metrics.loc[(df_patch_metrics['category_id'] == 12) | (df_patch_metrics['category_id'] == 13), 'category_name'] = 'Patch with core'
    # df_patch_metrics.loc[(df_patch_metrics['category_id'] == 12) | (df_patch_metrics['category_id'] == 13), 'category_id'] = 13

    # Save the patch metrics
    if save_csv:
        filename = os.path.join(outdir, f'{stub}_patch_metrics.csv')
        df_patch_metrics.to_csv(filename, index=False)
        print("Saved:", filename)
    
    # Assign linear and non-linear categories
    da_linear = da_filtered
    for i, row in df_patch_metrics.iterrows():
        if row["category_id"] == 14:  # currently uncategorised
            
            if row["skeleton_length"] > min_shelterbelt_length and row["skeleton_width"] < max_shelterbelt_width:
                new_class = 18  # linear features
            else:
                new_class = 19  # non-linear features
    
            mask = (assigned_labels == row["label"])
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
            label_to_category = dict(zip(df_patch_metrics['label'], df_patch_metrics['category_id']))  # I don't think this done anything now that I have to split_disconnected_clusters before skeleton_stats
            mapped_categories = np.vectorize(lambda x: label_to_category.get(x, 11))(label_ids)

        da_linear.data[remaining_mask] = mapped_categories
    
    ds = da_linear.to_dataset(name="linear_categories")
    ds['labelled_categories'] = (["y", "x"], assigned_labels)  # just so the output ds has both
    
    # Save the da_linear tif
    if save_tif:
        filename_linear = os.path.join(outdir, f'{stub}_linear_categories.tif')
        tif_categorical(ds['linear_categories'], filename_linear, linear_categories_cmap) 
    
    if plot:
        filename = os.path.join(outdir, f'{stub}_linear_categories.png')  # For creating a png output
        # filename = None  # For debugging within the notebook
        visualise_categories(da_linear, filename, linear_categories_cmap, linear_categories_labels, "Linear Categories")
    
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


# +
# if __name__ == '__main__':

#     args = parse_arguments()

#     geotif = args.buffer_tif
#     outdir = args.outdir
#     stub = args.stub
#     ds, df = patch_metrics(geotif, outdir, stub)

#     geotif = os.path.join(outdir, f"{stub}_linear_categories.tif")
#     dfs = class_metrics(geotif, outdir, stub)
# -

# Running locally
outdir = "../../../outdir/"
buffer_tif = "../../../outdir/34_37-148_42_y2018_predicted_buffer_categories.tif"
min_patch_size = 20
min_branch_length = min_patch_size
stub="TEST2"
save_csv=True
plot = False
save_tif=True
save_labels=True
# stub = "shelter_indices"
# geotif = os.path.join(outdir, f"{stub}_buffer_categories.tif")
# outdir="/scratch/xe2/cb8590"

# patch_metrics("../../../outdir/hydrolines_buffer_categories.tif")
# patch_metrics("/scratch/xe2/cb8590/tmp/34_37-148_42_y2018_predicted_buffer_categories.tif", outdir=outdir, plot=True)
df = patch_metrics(buffer_tif, stub=stub, outdir=outdir, plot=True)


