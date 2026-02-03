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
# # Shelter Categories Demo
#
# Minimal demo of `shelter_categories` with a small test dataset.

# %% [markdown]
# ## Setup

# %%
from shelterbelts.utils import create_test_woody_veg_dataset, visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories, shelter_categories_cmap, shelter_categories_labels

# Load test data and create tree categories
# (using in-memory datasets to avoid file I/O)
ds_input = create_test_woody_veg_dataset()
ds_cat = tree_categories(ds_input, stub='demo', outdir='/tmp', plot=False, save_tif=False)

# %% [markdown]
# ## Default parameters

# %%
ds_default = shelter_categories(ds_cat, outdir='/tmp', stub='default', plot=False, savetif=False)
visualise_categories(
    ds_default['shelter_categories'],
    colormap=shelter_categories_cmap,
    labels=shelter_categories_labels
)

# %% [markdown]
# ## Parameter: distance_threshold

# %%
ds_dist10 = shelter_categories(ds_cat, outdir='/tmp', stub='dist10', plot=False, savetif=False, distance_threshold=10)
ds_dist30 = shelter_categories(ds_cat, outdir='/tmp', stub='dist30', plot=False, savetif=False, distance_threshold=30)

visualise_categories_sidebyside(
    ds_dist10['shelter_categories'], ds_dist30['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="distance_threshold=10", title2="distance_threshold=30"
)

# %% [markdown]
# ## Parameter: density_threshold

# %%
ds_den3 = shelter_categories(ds_cat, outdir='/tmp', stub='den3', plot=False, savetif=False, density_threshold=3)
ds_den10 = shelter_categories(ds_cat, outdir='/tmp', stub='den10', plot=False, savetif=False, density_threshold=10)

visualise_categories_sidebyside(
    ds_den3['shelter_categories'], ds_den10['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="density_threshold=3", title2="density_threshold=10"
)
