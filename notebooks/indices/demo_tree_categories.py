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
# This notebook explores how each parameter affects the categorization of woody vegetation.

# %% [markdown]
# ## Setup

# %%
import numpy as np

from shelterbelts.utils import create_test_woody_veg_dataset, visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.tree_categories import tree_categories, tree_categories_cmap, tree_categories_labels

# Load test data
ds_input = create_test_woody_veg_dataset()
print(f"Input dimensions: {ds_input['woody_veg'].shape}")

# %% [markdown]
# ## Default Parameters
#
# First, let's run with default parameters to see the baseline categorization:

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
# - **Low value (1)**: Thin edge zones, more core area
# - **High value (5)**: Thick edge zones, less core area

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
# - **Low value (10)**: More small clusters classified as patches
# - **High value (30)**: Only larger clusters classified as patches

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
# - **Low value (0)**: No gap bridging, more fragmented patches
# - **High value (2)**: Bridges small gaps, more connected patches

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
# The `strict_core_area` parameter controls whether core areas must be strictly connected.
#
# - **False**: Relaxed connectivity, allows patchy core areas
# - **True**: Strict connectivity, requires fully connected core areas

# %%
ds_strict_false = tree_categories(ds_input, stub='strict_false', outdir='/tmp', plot=False, save_tif=False, strict_core_area=False)
ds_strict_true = tree_categories(ds_input, stub='strict_true', outdir='/tmp', plot=False, save_tif=False, strict_core_area=True)

visualise_categories_sidebyside(
    ds_strict_false['tree_categories'], ds_strict_true['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="strict_core_area=False", title2="strict_core_area=True"
)

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how each parameter affects tree categorization:
# - `edge_size`: Controls the width of edge zones
# - `min_patch_size`: Sets minimum size for patches vs scattered trees
# - `max_gap_size`: Determines gap bridging for connectivity
# - `strict_core_area`: Enforces core area connectivity rules
