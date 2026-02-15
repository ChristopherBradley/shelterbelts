# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tree Categories Demo
#
# Demonstrates the `tree_categories` function with different parameter values.

# %% [markdown]
# ## Setup

# %%
import rioxarray as rxr

from shelterbelts.utils.visualization import visualise_categories_sidebyside, visualise_categories
from shelterbelts.utils.filepaths import get_filename as get_example_data
from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.tree_categories import tree_categories_cmap, tree_categories_labels

# Load test data
test_file = get_example_data('g2_26729_binary_tree_cover_10m.tiff')
da_trees = rxr.open_rasterio(test_file).isel(band=0).drop_vars('band')
ds_input = da_trees.to_dataset(name='woody_veg')
print(f"Input dimensions: {ds_input['woody_veg'].shape}")

# %% [markdown]
# ## Default Parameters
#
# First, let's run with default parameters:

# %%
ds_default = tree_categories(ds_input, stub='default', outdir='/tmp', plot=False, save_tif=False)
print(f"Output variables: {set(ds_default.data_vars)}")

# %%
visualise_categories(
    ds_default['tree_categories'],
    colormap=tree_categories_cmap,
    labels=tree_categories_labels
)

# %% [markdown]
# ## Parameter: edge_size
#
# The `edge_size` parameter defines the distance (in pixels) from the edge of a patch.
# Areas beyond this distance from edges are classified as "Patch Core".
#
# - **Low value (1)**: Thin edges
# - **High value (5)**: Thick edges

# %%
ds_edge1 = tree_categories(ds_input, stub='edge1', outdir='/tmp', plot=False, save_tif=False, edge_size=1)
ds_edge5 = tree_categories(ds_input, stub='edge5', outdir='/tmp', plot=False, save_tif=False, edge_size=5)

visualise_categories_sidebyside(
    ds_edge1['tree_categories'], ds_edge5['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="edge_size=1", title2="edge_size=5"
)

# %% [markdown]
# ## Parameter: min_patch_size
#
# The `min_patch_size` parameter sets the minimum area (in pixels) for a cluster to be
# considered a patch rather than scattered trees.
#
# - **Low value (10)**: Less scattered trees
# - **High value (30)**: More scattered trees

# %%
ds_patch10 = tree_categories(ds_input, stub='patch10', outdir='/tmp', plot=False, save_tif=False, min_patch_size=10)
ds_patch30 = tree_categories(ds_input, stub='patch30', outdir='/tmp', plot=False, save_tif=False, min_patch_size=30)

visualise_categories_sidebyside(
    ds_patch10['tree_categories'], ds_patch30['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="min_patch_size=10", title2="min_patch_size=30"
)

# %% [markdown]
# ## Parameter: max_gap_size
#
# The `max_gap_size` parameter determines the maximum gap (in pixels) that can be bridged
# when connecting tree clusters into patches.
#
# - **Low value (0)**: More scattered trees and smaller patches
# - **High value (2)**: Less scattered trees and larger patches

# %%
ds_gap0 = tree_categories(ds_input, stub='gap0', outdir='/tmp', plot=False, save_tif=False, max_gap_size=0)
ds_gap2 = tree_categories(ds_input, stub='gap2', outdir='/tmp', plot=False, save_tif=False, max_gap_size=2)

visualise_categories_sidebyside(
    ds_gap0['tree_categories'], ds_gap2['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="max_gap_size=0", title2="max_gap_size=2"
)

# %% [markdown]
# ## Parameter: strict_core_area
#
# The `strict_core_area` parameter changes the method for defining core areas.
#
# - **False**: Use dilation and erosion to allow some irregularity.
# - **True**: Enforce that core areas exceed the edge_size at all points.

# %%
ds_strict_false = tree_categories(ds_input, stub='strict_false', outdir='/tmp', plot=False, save_tif=False, strict_core_area=False)
ds_strict_true = tree_categories(ds_input, stub='strict_true', outdir='/tmp', plot=False, save_tif=False, strict_core_area=True)

visualise_categories_sidebyside(
    ds_strict_false['tree_categories'], ds_strict_true['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="strict_core_area=False", title2="strict_core_area=True"
)
