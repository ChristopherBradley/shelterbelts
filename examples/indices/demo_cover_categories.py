# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cover Categories Demo
#
# Demonstrates the `cover_categories` function, which overlays WorldCover land cover types onto the tree categories.

# %% [markdown]
# ## Setup

# %%
import rioxarray as rxr

from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories
from shelterbelts.indices.cover_categories import cover_categories, cover_categories_cmap, cover_categories_labels

# Example data
tree_file = get_filename('g2_26729_tree_categories.tif')
worldcover_file = get_filename('g2_26729_worldcover.tif')

# %% [markdown]
# ## Default Parameters

# %%
ds = cover_categories(tree_file, worldcover_file)
ds

# %%
visualise_categories(
    ds['cover_categories'],
    colormap=cover_categories_cmap,
    labels=cover_categories_labels,
    title="Default"
)

# %% [markdown]
# ## Using Datasets as input

# %%
da_tree = rxr.open_rasterio(tree_file).squeeze('band').drop_vars('band')
da_worldcover = rxr.open_rasterio(worldcover_file).squeeze('band').drop_vars('band')

ds2 = cover_categories(da_tree, da_worldcover)
visualise_categories(
    ds2['cover_categories'],
    colormap=cover_categories_cmap,
    labels=cover_categories_labels
)

# %% [markdown]
# ## Command Line Interface

# %%
# !python -m shelterbelts.indices.cover_categories --help

# %%
# !python -m shelterbelts.indices.cover_categories {tree_file} {worldcover_file}

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# # !rm *.tif
# # !rm *.png
# # !rm *.xml  # These get generated if you load the tifs in QGIS
