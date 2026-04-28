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
# # LiDAR Processing
#
# ``lidar.py`` takes a .laz file and generates a canopy height model and percent cover raster (or binary tif), which can be used as input to the rest of the shelter pipeline. You can download .laz files for any location in NSW at [ELVIS](https://elevation.fsdf.org.au/).

# %%
from shelterbelts.classifications.lidar import lidar
from shelterbelts.utils.filepaths import laz_sample

# %% [markdown]
# ## Generating a binary raster using existing 'Tall Vegetation' classifications
#

# %%
counts, da_tree = lidar(
    laz_sample,
    category5=True,
    binary=True,
    resolution=1
)

# %%
counts.plot()  # Number of points with the classification 'Tall Vegetation' in each pixel

# %%
da_tree.plot() # binary tree raster

# %% [markdown]
# ## Generating a Canopy Height Model with pdal's hag_nn algorithm

# %%
chm, da_tree = lidar(
    laz_sample,
    resolution=1
)

# %%
chm.plot() # Maximum height of a point in each pixel

# %%
da_tree.plot()  # percent cover raster (but each pixel is 100% tree cover)

# %% [markdown]
# ## Changing the height threshold

# %%
chm, da_tree = lidar(
    laz_sample,
    resolution=1,
    height_threshold=10 # metres
)
da_tree.plot()

# %% [markdown]
# ## Changing the output resolution

# %%
chm, da_tree = lidar(
    laz_sample,
    resolution=5 # metres
)
da_tree.plot() # Percent cover raster (edges now have only partial tree cover)
