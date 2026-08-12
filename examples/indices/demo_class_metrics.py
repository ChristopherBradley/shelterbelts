# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-widgets,-varInspector,-jupytext.text_representation.jupytext_version
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
# # Class Metrics Demo
#
# `class_metrics` calculates the percentage cover in each category from a shelter_categories output,
# summarised across four sheets: Overall, Landcover, Trees, and Shelter.

# %%
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.indices.class_metrics import class_metrics

# Example data: a shelter_categories output (the last pipeline step), so the Shelter sheet
# reflects the sheltered vs unsheltered split by tree type
shelter_file = get_filename('g2_26729_shelter_categories.tif')

# %% [markdown]
# ## Class Metrics

# %%
dfs = class_metrics(shelter_file)

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
# !python -m shelterbelts.indices.class_metrics {shelter_file} --stub command_line

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# # !rm *.xlsx
