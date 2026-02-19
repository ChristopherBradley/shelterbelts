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
# # Catchments Demo
#
# Demonstrates the `catchments` function, which generates gully and ridge rasters
# from a Digital Elevation Model (DEM) using pysheds.

# %% [markdown]
# ## Setup

# %%
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.indices.catchments import catchments, plot_catchments

# Example data
dem_file = get_filename('g2_26729_DEM-H.tif')

# %% [markdown]
# ## Default Parameters

# %%
ds = catchments(dem_file)
ds

# %%
plot_catchments(ds)

# %% [markdown]
# ## Changing num_catchments
#
# The `num_catchments` parameter controls how many catchments are identified.

# %%
ds5 = catchments(dem_file, stub='num_catchments5', num_catchments=5)
ds20 = catchments(dem_file, stub='num_catchments20', num_catchments=20)

# %%
# Can we have two side-by-side plots here using the existing plot_catchments function?
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ds5['gullies'].plot(ax=axes[0], cmap='Blues')
axes[0].set_title('num_catchments=5')
ds20['gullies'].plot(ax=axes[1], cmap='Blues')
axes[1].set_title('num_catchments=20')
plt.tight_layout()

# %% [markdown]
# ## Command Line Interface

# %%
# !python -m shelterbelts.indices.catchments --help

# %%
