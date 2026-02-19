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
# # Cover Categories Demo
#
# Demonstrates the `cover_categories` function, which combines shelter categories with WorldCover land cover types.

# %% [markdown]
# ## Setup

# %%
import rioxarray as rxr

from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories
from shelterbelts.indices.cover_categories import cover_categories, cover_categories_cmap, cover_categories_labels

# Example data
shelter_file = get_filename('g2_26729_shelter_categories.tif')
worldcover_file = get_filename('g2_26729_worldcover.tif')

# %% [markdown]
# ## Default Parameters

# %%
ds = cover_categories(shelter_file, worldcover_file)
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
da_shelter = rxr.open_rasterio(shelter_file).squeeze('band').drop_vars('band')
da_worldcover = rxr.open_rasterio(worldcover_file).squeeze('band').drop_vars('band')

ds2 = cover_categories(da_shelter, da_worldcover)
visualise_categories(
    ds2['cover_categories'],
    colormap=cover_categories_cmap,
    labels=cover_categories_labels
)

# %% [markdown]
# ## Command Line Interface

# %%
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

# %%
# !python -m shelterbelts.indices.cover_categories --help

# %%
# !python -m shelterbelts.indices.cover_categories {shelter_file} {worldcover_file}

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# !rm ../notebooks/indices/*.tif
# !rm ../notebooks/indices/*.png
# !rm ../notebooks/indices/*.xml  # These get generated if you load the tifs in QGIS
