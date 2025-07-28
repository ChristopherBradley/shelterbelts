import os
import numpy as np
import pandas as pd
import rioxarray as rxr

from shelterbelts.indices.buffer_categories import buffer_categories_cmap, buffer_categories_labels


def class_metrics(geometry, folder):
    """Calculate the percentage cover in each class.
    
    Parameters
    ----------
        geometry: The region to calculate metrics
        folder: The folder containing tif files with shelter categories for that region
            
    Returns
    -------
        dict_metrics: A dictionary with 4 dataframes: overall, landcover, trees, shelter

    Downloads
    ---------
        class_metrics.xlsx: An excel file with each dataframe in a separate tab.

    """


def patch_metrics(geometry, folder):
    """Calculate the length, width, height, direction, area, perimeter for each patch.
        Also calculates an overall mean and standard deviation for each attribute across all patches.
    
    Parameters
    ----------
        geometry: The region to calculate metrics
        folder: The folder containing tif files with shelter categories for that region
            
    Returns
    -------
        df_individual: Individual attributes for each patch.
        df_aggregates: Aggregated attributes for all patches.

    Downloads
    ---------
        patch_metrics_individual.csv
        patch_metrics_aggregated.csv

    """



# Read in the buffer categories
outdir = "../../../outdir/"
stub = "g2_26729"
buffer_tif = os.path.join(outdir, f"{stub}_buffer_categories.tif")


da = rxr.open_rasterio(buffer_tif).isel(band=0)

# +
# Overall stats per category
counts = da.values.ravel()
counts = pd.Series(counts).value_counts().sort_index()
total_pixels = da.shape[0] * da.shape[1]
# total_pixels = np.isfinite(da.values).sum() # Could use this if I need to worry about NaN values

# Construct a dataframe
df_overall = pd.DataFrame({
    'category_id': counts.index,
    'label': [buffer_categories_labels.get(cat, 'Unknown') for cat in counts.index],
    'pixel_count': counts.values,
    'percentage': (counts.values / total_pixels) * 100
})
df_overall = df_overall.set_index('label')
# df_overall = df_overall.sort_values(by='percentage', ascending=False)

df_overall

# +
# Landcover groups
landcover_groups = {
    10: "Trees",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    80: "Water",
}
def group_label(cat_id):
    """Apply broader group categories to each pixel"""
    return landcover_groups.get((cat_id // 10) * 10, 'Other')  # Flooring to the nearest 10

# Assign a group to each category
df_overall['landcover_group'] = df_overall['category_id'].apply(group_label)

df_grouped = df_overall.groupby('landcover_group')[['pixel_count', 'percentage']].sum()
df_grouped = df_grouped.sort_values(by='percentage', ascending=False)
df_grouped['percentage'] = df_grouped['percentage'].round(2)

df_grouped

# +
# Tree groups
df_trees = df_overall[df_overall['landcover_group'] == 'Trees'].copy()
total_tree_pixels = df_trees['pixel_count'].sum()
df_trees['percentage'] = (df_trees['pixel_count'] / total_tree_pixels) * 100
df_trees['percentage'] = df_trees['percentage'].round(2)
df_trees = df_trees.drop(columns=['category_id', 'landcover_group'])
df_trees = df_trees.sort_values(by='percentage', ascending=False)

df_trees
# -



# +
# Shelter class stats
df_production = df_overall[df_overall.index.str.contains("sheltered", case=False)].copy()
df_production['shelter_status'] = df_production.index.str.split().str[0] # First word is the shelter status
df_production['production_category'] = df_production.index.str.split().str[1]  # Last word is Grassland or Cropland

df_summary = df_sel.pivot_table(
    index='production_category',
    columns='shelter_status',
    values='pixel_count',
    aggfunc='sum',
    fill_value=0
)
df_summary.loc['Total'] = df_summary.sum()
df_percent = df_summary.div(df_summary.sum(axis=1), axis=0) * 100
df_percent = df_percent.round(2)

# -

df_sel = df_production

# +
# Pivot to create summary of pixel counts

df_percent
# -



# +
# Do the class metrics
# -



# +
# Re-read masters thesis & R package
# -

# Reuse the scipy label function, and do more with each patch. MVP is just to have at least some stats per patch.



