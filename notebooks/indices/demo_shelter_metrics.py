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
# # Shelter Metrics Demo
#
# Demonstrates `patch_metrics` and `class_metrics` from `shelter_metrics.py`.
# - `patch_metrics` classifies tree clusters as linear (shelterbelts) vs non-linear (patches)
# - `class_metrics` calculates percentage cover in each category

# %% [markdown]
# ## Setup

# %%
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.shelter_metrics import patch_metrics, class_metrics
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels

# Example data
buffer_file = get_filename('g2_26729_buffer_categories.tif')

# %% [markdown]
# ## Default Parameters

# %%
ds, df = patch_metrics(buffer_file, outdir='/tmp', stub='default', plot=False, save_csv=False, save_tif=False, save_labels=False)
print(f"Output variables: {set(ds.data_vars)}")
print(f"Number of patches: {len(df)}")

# %%
visualise_categories(
    ds['linear_categories'],
    colormap=linear_categories_cmap,
    labels=linear_categories_labels
)

# %% [markdown]
# ## Patch metrics DataFrame
#
# Each row represents a tree cluster with its geometric properties.

# %%
df.head(10)

# %% [markdown]
# ## Changing min_shelterbelt_length and max_shelterbelt_width
#
# These parameters control which clusters are classified as linear (shelterbelts).

# %%
ds1, _ = patch_metrics(buffer_file, outdir='/tmp', stub='short', plot=False, save_csv=False, save_tif=False, save_labels=False,
                        min_shelterbelt_length=10, max_shelterbelt_width=4)
ds2, _ = patch_metrics(buffer_file, outdir='/tmp', stub='long', plot=False, save_csv=False, save_tif=False, save_labels=False,
                        min_shelterbelt_length=25, max_shelterbelt_width=8)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="length=10, width=4", title2="length=25, width=8"
)

# %% [markdown]
# ## Class Metrics
#
# `class_metrics` calculates the percentage cover in each category.

# %%
linear_file = get_filename('g2_26729_linear_categories.tif')
dfs = class_metrics(linear_file, outdir='/tmp', stub='test', save_excel=False)

# %%
# Overall category breakdown
dfs['Overall']

# %%
# Landcover summary
dfs['Landcover']

# %%
# Tree category breakdown
dfs['Trees']

# %%
# Shelter statistics
dfs['Shelter']

# %% [markdown]
# ## Command Line Interface

# %%
# !python -m shelterbelts.indices.shelter_metrics --help
