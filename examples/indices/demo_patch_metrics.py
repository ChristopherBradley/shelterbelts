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
# # Patch Metrics Demo
#
# `patch_metrics` classifies tree clusters as linear (shelterbelts) vs non-linear (patches) and outputs
# the skeleton length/width statistics for each patch.

# %%
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.patch_metrics import patch_metrics
from shelterbelts.indices.shelter_categories import shelter_categories_cmap, shelter_categories_labels

# Example data
buffer_file = get_filename('g2_26729_gullies_and_roads_buffer_categories.tif')

# %% [markdown]
# ## Default Parameters

# %%
ds, df = patch_metrics(buffer_file)

# %%
df.head()

# %%
visualise_categories(
    ds['linear_categories'],
    colormap=shelter_categories_cmap,
    labels=shelter_categories_labels
)

# %% [markdown]
# ## Changing min_shelterbelt_length and max_shelterbelt_width

# %%
ds1, _ = patch_metrics(buffer_file, max_shelterbelt_width=8)
ds2, _ = patch_metrics(buffer_file, min_shelterbelt_length=25)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="max width=8", title2="min length=25"
)

# %% [markdown]
# ## Command Line Interface

# %%
# !python -m shelterbelts.indices.patch_metrics --help

# %%
# !python -m shelterbelts.indices.patch_metrics {buffer_file} --stub command_line

# %%
# !python -m shelterbelts.indices.patch_metrics {buffer_file} --min_shelterbelt_length 25 --max_shelterbelt_width 8 --stub command_line

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# # !rm *.tif
# # !rm *.png
# # !rm *.xml  # These get generated if you load the tifs in QGIS
# # !rm *.csv
