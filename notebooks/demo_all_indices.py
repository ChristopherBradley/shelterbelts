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
# # All Indices Demo
#
# This notebook demonstrates the full indices pipeline:  
# tree_categories → shelter_categories → cover_categories → buffer_categories → patch_metrics

# %%
from shelterbelts.utils.filepaths import get_filename, setup_repo_path
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.all_indices import indices_tif, indices_tifs
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels, class_metrics

setup_repo_path(subdir='')

# # Example data
# tree_cover_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
# tree_cover_file='/g/data/xe2/cb8590/Nick_Aus_treecover_10m/g1_02060_binary_tree_cover_10m.tiff'

# %%
tree_cover_file='/scratch/xe2/cb8590/Nick_Aus_treecover_10m/subfolder_8/g2_26729_binary_tree_cover_10m.tiff'
tree_cover_folder='/scratch/xe2/cb8590/Nick_Aus_treecover_10m/subfolder_1'


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
folder = f'data/multiple_binary_tifs'
indices_tifs(folder, suffix='tiff', limit=2)

# %%
# !ls {folder}

# %%
