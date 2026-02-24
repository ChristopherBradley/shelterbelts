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
# # Buffer Categories Demo

# %% [markdown]
# ## Setup

# %%
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.buffer_categories import buffer_categories_cmap, buffer_categories_labels

# Example data
cover_file = get_filename('g2_26729_cover_categories.tif')
gullies_file = get_filename('g2_26729_DEM-S_gullies.tif')
ridges_file = get_filename('g2_26729_DEM-S_ridges.tif')
roads_file = get_filename('g2_26729_roads.tif')

# %% [markdown]
# ## Default Parameters (gullies only)

# %%
ds_default = buffer_categories(cover_file, gullies_file)
ds_default

# %%
visualise_categories(
    ds_default['buffer_categories'],
    colormap=buffer_categories_cmap,
    labels=buffer_categories_labels
)

# %% [markdown]
# ## Changing the buffer_width
#
# The `buffer_width` parameter sets how many pixels away from a feature (gully, ridge, road) still count as within the buffer.

# %%
ds1 = buffer_categories(cover_file, gullies_file, buffer_width=1)
ds2 = buffer_categories(cover_file, gullies_file, buffer_width=5)
visualise_categories_sidebyside(
    ds1['buffer_categories'], ds2['buffer_categories'],
    colormap=buffer_categories_cmap, labels=buffer_categories_labels,
    title1="buffer_width=1", title2="buffer_width=5"
)

# %% [markdown]
# ## Adding ridges and roads
#
# Providing `ridges_data` and `roads_data` adds ridge buffer and road buffer categories.

# %%
ds_gullies = buffer_categories(cover_file, gullies_file, roads_data=roads_file, stub='gullies_and_roads')
ds_all = buffer_categories(cover_file, gullies_file, ridges_data=ridges_file, roads_data=roads_file, stub='all_buffers')
visualise_categories_sidebyside(
    ds_gullies['buffer_categories'], ds_all['buffer_categories'],
    colormap=buffer_categories_cmap, labels=buffer_categories_labels,
    title1="Gullies + Roads", title2="Gullies + Ridges + Roads"
)

# %% [markdown]
# ## Command Line Interface

# %%
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

# %%
# !python -m shelterbelts.indices.buffer_categories --help

# %%
# !python -m shelterbelts.indices.buffer_categories {cover_file} {gullies_file} --roads_data {roads_file} --buffer_width 4 --outdir ../notebooks/indices --stub command_line

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# !rm ../notebooks/indices/*.tif
# !rm ../notebooks/indices/*.png
# !rm ../notebooks/indices/*.xml  # These get generated if you load the tifs in QGIS
