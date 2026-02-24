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
# tree_categories → shelter_categories → cover_categories → buffer_categories → patch_metrics

# %%
from shelterbelts.utils.filepaths import get_filename, setup_repo_path
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.all_indices import indices_tif
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels, class_metrics

setup_repo_path(subdir='')

# Example data
tree_cover_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')

# %% [markdown]
# ## Default Parameters

# %%
# %%time
ds, df = indices_tif(tree_cover_file)

# %%
visualise_categories(
    ds['linear_categories'],
    colormap=linear_categories_cmap,
    labels=linear_categories_labels
)

# %%
df.head()

# %% [markdown]
# ## Changing min_patch_size
#
# In this example some of the smaller groups of trees are "Non-linear Patches" on the left and "Scattered Trees" on the right.

# %%
ds1, _ = indices_tif(tree_cover_file, stub='min_patch_size10', min_patch_size=10)
ds2, _ = indices_tif(tree_cover_file, stub='min_patch_size40', min_patch_size=40)
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
# In this example the riparian zone & road zone are 3 pixels wide (1 pixel buffer on each side) on the left, and 11 pixels wide (5 + 1 + 5) on the right.

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
# In this example there is just 1 Linear Patch on the left because the others don't meet the requirement of 25 pixels long. In addition, there is a 4th (wider) Linear Patch on the right compared to just 3 Linear Patches with the default values.

# %%
ds1, _ = indices_tif(tree_cover_file, outdir='/tmp', stub='length25', min_shelterbelt_length=25, buffer_width=5)
ds2, _ = indices_tif(tree_cover_file, outdir='/tmp', stub='width8', max_shelterbelt_width=8, buffer_width=5)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="min_shelterbelt_length=25", title2="max_shelterbelt_width=8"
)

# %% [markdown]
# ## Class Metrics

# %%
dfs = class_metrics(ds)

# %%
dfs['Overall']

# %%
dfs['Landcover']

# %%
dfs['Trees']

# %%
dfs['Shelter']

# %% [markdown]
# ## Command Line Interface

# %%
# !python -m shelterbelts.indices.all_indices --help
