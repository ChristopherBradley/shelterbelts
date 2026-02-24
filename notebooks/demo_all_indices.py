# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # All Indices Demo
#
# This notebook demonstrates the full indices pipeline:  
# tree_categories → shelter_categories → cover_categories → buffer_categories → patch_metrics & class_metrics

# %%
from shelterbelts.utils.filepaths import get_filename, setup_repo_path
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.all_indices import indices_tif
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels

setup_repo_path(subdir='')

# Example data
tree_cover_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')

# %% [markdown]
# ## Default Parameters

# %%
# %%time
ds, df = indices_tif(tree_cover_file)

# %%
df.head()

# %%
visualise_categories(
    ds['linear_categories'],
    colormap=linear_categories_cmap,
    labels=linear_categories_labels
)

# %% [markdown]
# ## Changing min_patch_size
#
# In this example some of the smaller groups of trees are "Non-linear Patches" on the left and "Scattered Trees" on the right.

# %%
ds1, _ = indices_tif(tree_cover_file, outdir='/tmp', stub='patch10', min_patch_size=10)
ds2, _ = indices_tif(tree_cover_file, outdir='/tmp', stub='patch40', min_patch_size=40)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="min_patch_size=10", title2="min_patch_size=40"
)

# %% [markdown]
# ## Changing edge_size
#
# In this example the Patch Edge around the Patch Core is 1 pixel wide on the left, and 5 pixels wide on the right.

# %%
ds1, _ = indices_tif(tree_cover_file, stub='edge_size1', edge_size=1)
ds2, _ = indices_tif(tree_cover_file, stub='edge_size5', edge_size=5)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="edge_size=1", title2="edge_size=5"
)

# %% [markdown]
# ## Changing buffer_width
#
# In this example the riparian zone & road zone are 3 pixels wide (1 pixel buffer on each side) on the left, and 

# %%
# !ls

# %%
ds1, _ = indices_tif(tree_cover_file, stub='buffer1', buffer_width=1)
ds2, _ = indices_tif(tree_cover_file, stub='buffer5', buffer_width=5)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="buffer_width=1", title2="buffer_width=5"
)

# %% [markdown]
# ## Changing min_shelterbelt_length and max_shelterbelt_width
#
# These parameters control the classification of tree clusters as linear (shelterbelts) vs non-linear (patches).

# %%
ds1, _ = indices_tif(tree_cover_file, outdir='/tmp', stub='length25', min_shelterbelt_length=25)
ds2, _ = indices_tif(tree_cover_file, outdir='/tmp', stub='width8', max_shelterbelt_width=8)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="min_shelterbelt_length=25", title2="max_shelterbelt_width=8"
)

# %% [markdown]
# ## Command Line Interface

# %%
# !python -m shelterbelts.indices.all_indices --help

# %% [markdown]
# ### Cleanup

# %%
# !rm /tmp/gap*.tif /tmp/patch*.tif /tmp/edge*.tif /tmp/length*.tif /tmp/width*.tif /tmp/dist*.tif /tmp/buf*.tif 2>/dev/null
# !rm /tmp/gap*.png /tmp/patch*.png /tmp/edge*.png /tmp/length*.png /tmp/width*.png /tmp/dist*.png /tmp/buf*.png 2>/dev/null
# !rm /tmp/gap*.csv /tmp/patch*.csv /tmp/edge*.csv /tmp/length*.csv /tmp/width*.csv /tmp/dist*.csv /tmp/buf*.csv 2>/dev/null
# !rm /tmp/gap*.xlsx /tmp/patch*.xlsx /tmp/edge*.xlsx /tmp/length*.xlsx /tmp/width*.xlsx /tmp/dist*.xlsx /tmp/buf*.xlsx 2>/dev/null
