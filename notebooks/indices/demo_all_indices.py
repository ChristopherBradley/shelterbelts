# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # All Indices Demo
#
# Demonstrates the `indices_tif` function, which runs the complete pipeline:
# tree_categories → shelter_categories → cover_categories → buffer_categories → patch_metrics
#
# This is the main entry point for processing a single percent-cover GeoTIFF.

# %% [markdown]
# ## Setup

# %%
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories
from shelterbelts.indices.all_indices import indices_tif
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels

# Example data - a binary tree cover tif
tree_cover_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')

# %% [markdown]
# ## Default Parameters
# Runs the full pipeline with default settings (no wind data).

# %%
# %%time
ds, df = indices_tif(tree_cover_file, outdir='/tmp', stub='demo_defaults', debug=False)
print(f"Output variables: {set(ds.data_vars)}")
print(f"Number of patches: {len(df)}")

# %%
visualise_categories(
    ds['linear_categories'],
    colormap=linear_categories_cmap,
    labels=linear_categories_labels
)

# %% [markdown]
# ## Changing tree classification parameters
#
# These parameters affect how tree pixels are grouped and categorised.

# %%
# More aggressive gap bridging and smaller patch threshold
ds2, df2 = indices_tif(tree_cover_file, outdir='/tmp', stub='demo_gaps',
                        max_gap_size=2, min_patch_size=10, debug=False)

visualise_categories(
    ds2['linear_categories'],
    colormap=linear_categories_cmap,
    labels=linear_categories_labels,
    title="max_gap_size=2, min_patch_size=10"
)

# %% [markdown]
# ## Command Line Interface

# %%
# !python -m shelterbelts.indices.all_indices --help

# %% [markdown]
# ### Cleanup

# %%
# !rm /tmp/demo_*.tif /tmp/demo_*.png /tmp/demo_*.csv /tmp/demo_*.xlsx 2>/dev/null
