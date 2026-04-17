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
# # Catchments Demo
#
# Demonstrates the `catchments` function, which generates gully and ridge rasters
# from a Digital Elevation Model (DEM) using pysheds.

# %% [markdown]
# ## Setup

# %%
from shelterbelts.indices.catchments import catchments
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import plot_catchments, plot_catchments_sidebyside

# Example data
dem_file = get_filename('g2_26729_DEM-H.tif')

# %% [markdown]
# ## Default Parameters

# %%
ds = catchments(dem_file)
ds

# %%
plot_catchments(ds, title="num_catchments=10")

# %% [markdown]
# ## Changing num_catchments
#
# The `num_catchments` parameter controls how many catchments are identified.

# %%
ds5 = catchments(dem_file, stub='num_catchments5', num_catchments=5)
ds20 = catchments(dem_file, stub='num_catchments20', num_catchments=20)

# %%
plot_catchments_sidebyside(ds5, ds20, title1='num_catchments=5', title2='num_catchments=20')

# %% [markdown]
# ## Command Line Interface

# %%
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

# %%
# !python -m shelterbelts.indices.catchments --help

# %%
# !python -m shelterbelts.indices.catchments {dem_file} --num_catchments 7 --stub command_line --outdir ../examples/indices

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# !rm ../examples/indices/*.tif
# !rm ../examples/indices/*.png
# !rm ../examples/indices/*.xml
