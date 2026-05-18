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
# # Merging GeoTIFF Tiles
#
# ``merge_tifs`` stitches a folder of GeoTIFF tiles into a single raster. The sample data in ``data/quartered_binary_tifs/`` contains four quarters of ``g2_26729_linear_categories.tif`` plus a fifth tile covering the same extent as the top-left quarter but with only binary categories, and named with an older date to demo the ``dedup`` functionality.

# %%
import rioxarray as rxr
import matplotlib.pyplot as plt

from shelterbelts.classifications.merge_tifs import merge_tifs
from shelterbelts.utils.filepaths import quartered_tifs_dir, _repo_root
from shelterbelts.utils.visualisation import visualise_categories
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels

# %%
# Listing the files to merge
# !ls {_repo_root / 'data' / 'quartered_linear_tifs'}/*.tif

# %% [markdown]
# ## Merging four tiles (no deduplication)
#
# In this first example, the top left quarter happens to use the '2019' (binary) tif over the '2020' categorical tif that we actually want.

# %%
da = merge_tifs(quartered_tifs_dir, suffix='.tif', dont_reproject=True)

# %%
visualise_categories(da, colormap=linear_categories_cmap, labels=linear_categories_labels,
                     title='Merged tifs (no deduplication)')

# %% [markdown]
# ## Merging with deduplication
#
# In this second example, we use ``dedup=True`` to choose the most recent tile when multiple tiles overlap (this is relevant when downloading from ELVIS and you want the most recent lidar aquisition)

# %%
da_dedup = merge_tifs(quartered_tifs_dir, tmpdir='tmpdir', suffix='.tif',
                      dont_reproject=True, dedup=True)

# %%
visualise_categories(da_dedup, colormap=linear_categories_cmap, labels=linear_categories_labels,
                     title='Merged tifs (with deduplication)')
