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
# # Opportunities Demo
#
# Demonstrates the `opportunities` function with different parameter values
# using real data files.

# %% [markdown]
# ## Setup

# %%
from shelterbelts.indices.opportunities import opportunities, opportunity_cmap, opportunity_labels
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories

tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
roads_file = get_filename('g2_26729_roads.tif')
gullies_file = get_filename('g2_26729_hydrolines.tif')
dem_file = get_filename('g2_26729_DEM-H.tif')
worldcover_file = get_filename('g2_26729_worldcover.tif')

print(f"Tree file: {tree_file}")

common = dict(dem_data=dem_file, worldcover_data=worldcover_file, outdir='/tmp', savetif=False, plot=False)

# %% [markdown]
# ## Default Parameters

# %%
ds_default = opportunities(
    tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, stub='demo_default',
)
ds_default

# %%
visualise_categories(
    ds_default['opportunities'],
    colormap=opportunity_cmap,
    labels=opportunity_labels
)

# %% [markdown]
# ## Just roads vs just gullies
#
# Showing the effect of each feature in isolation by zeroing out the other.

# %%
import rioxarray as rxr
da_zero = rxr.open_rasterio(roads_file).isel(band=0).drop_vars('band') * 0

ds_roads = opportunities(tree_file, roads_data=roads_file, gullies_data=da_zero, **common, stub='demo_roads', contour_spacing=0, width=5)
ds_gullies = opportunities(tree_file, roads_data=da_zero, gullies_data=gullies_file, **common, stub='demo_gullies', contour_spacing=0)

visualise_categories_sidebyside(
    ds_roads['opportunities'], ds_gullies['opportunities'],
    colormap=opportunity_cmap, labels=opportunity_labels,
    title1="Just roads", title2="Just gullies"
)

# %% [markdown]
# ## Changing the width
#
# The `width` parameter controls how many pixels away from each feature still
# count as within the buffer for planting opportunities.

# %%
ds_w1 = opportunities(
    tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, stub='demo_w1', width=1,
)
ds_w5 = opportunities(
    tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, stub='demo_w3', width=3,
)

visualise_categories_sidebyside(
    ds_w1['opportunities'], ds_w5['opportunities'],
    colormap=opportunity_cmap, labels=opportunity_labels,
    title1="width=1", title2="width=3"
)

# %% [markdown]
# ## Changing the contour spacing
#
# The `contour_spacing` parameter controls the number of pixels between each
# contour line. Smaller values produce more contour opportunities.

# %%
ds_cs5 = opportunities(
    tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, stub='demo_cs5', contour_spacing=10,
)
ds_cs20 = opportunities(
    tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, stub='demo_cs20', contour_spacing=20,
)

visualise_categories_sidebyside(
    ds_cs5['opportunities'], ds_cs20['opportunities'],
    colormap=opportunity_cmap, labels=opportunity_labels,
    title1="contour_spacing=5", title2="contour_spacing=20"
)

# %% [markdown]
# ## Command Line Interface
# You can also use the function from the command line with the same defaults and parameters.

# %%
# !python -m shelterbelts.indices.opportunities --help
