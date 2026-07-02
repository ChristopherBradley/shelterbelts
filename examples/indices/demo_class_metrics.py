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
# # Class Metrics Demo
#
# `class_metrics` calculates the percentage cover in each category from a shelter_categories output,
# summarised across four sheets: Overall, Landcover, Trees, and Shelter.

# %%
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.indices.class_metrics import class_metrics

# Example data
linear_file = get_filename('g2_26729_linear_categories.tif')

# %% [markdown]
# ## Class Metrics

# %%
dfs = class_metrics(linear_file)

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
# !python -m shelterbelts.indices.class_metrics --help

# %%
# !python -m shelterbelts.indices.class_metrics {linear_file} --stub command_line

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# # !rm *.xlsx
