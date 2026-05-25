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
# # Indices Latlon Demo
#
# Demonstrates `indices_latlon`, which runs the complete shelterbelts pipeline for any
# lat/lon location by auto-downloading all required data:
#
# - **Canopy height** — Meta/Tolan global 1 m CHM, binarised at `height_threshold` and
#   average-resampled to 10 m to give percent cover
# - **ESA WorldCover** — provides the 10 m reference grid and land-use categories
# - **Terrain tiles** — MapZen elevation used to derive gullies and ridge lines
# - **OpenStreetMap roads** — road network for the region of interest
# - **BARRA wind** — only downloaded when `wind_method` is set

# %%
from shelterbelts.indices.all_indices import indices_latlon
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels
from shelterbelts.utils.visualisation import visualise_categories, visualise_categories_sidebyside

# Default location: Milgadara, NSW, Australia
lat, lon, buffer = -34.389, 148.469, 0.01  # distance in degrees in each direction, so 0.01 is ~2km x 2km

# %% [markdown]
# ## Default Parameters
# * Note this may take a while (~10 mins) to run the first time at any given location since it needs to download a ~60MB tif file.

# %%
# %%time
ds, df = indices_latlon(lat, lon, buffer, debug=True)

# %%
df.head()

# %%
visualise_categories(
    ds['linear_categories'],
    colormap=linear_categories_cmap,
    labels=linear_categories_labels
)

# %% [markdown]
# ## Changing the buffer
#
# The `buffer` parameter sets the half-width of the region of interest in degrees
# (~1° latitude ≈ 111 km). Larger buffers take longer to run.

# %%
# %%time
ds1, _ = indices_latlon(lat, lon, buffer=0.02, stub='buffer_medium')
ds2, _ = indices_latlon(lat, lon, buffer=0.05, stub='buffer_large')
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="buffer=0.02 (~4 km)", title2="buffer=0.04 (~8 km)"
)

# %% [markdown]
# ## Changing the height_threshold
#
# The `height_threshold` (metres) controls which pixels in the 1 m canopy height model
# are classified as trees or not.

# %%
ds1, _ = indices_latlon(lat, lon, buffer, height_threshold=1.0, stub='height1')
ds2, _ = indices_latlon(lat, lon, buffer, height_threshold=8.0, stub='height8')
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="height_threshold=1 m", title2="height_threshold=8 m"
)

# %% [markdown]
# ## Changing the cover_threshold
#
# After resampling from 1 m to 10 m, each pixel holds the percentage of 1 m sub-pixels
# that were classified as trees. `cover_threshold` sets the minimum percentage required
# to count a 10 m pixel as a tree.

# %%
ds1, _ = indices_latlon(lat, lon, buffer, cover_threshold=1, stub='cover1')
ds2, _ = indices_latlon(lat, lon, buffer, cover_threshold=70, stub='cover70')
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="cover_threshold=1%", title2="cover_threshold=70%"
)

# %% [markdown]
# ## Using Wind Data
#
# Set `wind_method` to incorporate BARRA wind reanalysis data when determining shelter
# direction. The wind data is automatically downloaded for the region.
#
# - **MOST_COMMON**: shelter downwind of the most common strong-wind direction
# - **WINDWARD**: shelter both downwind (full distance) and upwind (half distance)

# %%
ds1, _ = indices_latlon(lat, lon, buffer, wind_method='MOST_COMMON', stub='wind_most_common')
ds2, _ = indices_latlon(lat, lon, buffer, wind_method='WINDWARD', stub='wind_windward')
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="wind_method=MOST_COMMON", title2="wind_method=WINDWARD"
)

# %% [markdown]
# ### Cleanup
# Remove output files created by this notebook

# %%
# # !rm ./*.png
# # !rm ./*.csv
# # !rm ./*.xml  # Generated if tifs are opened in QGIS
# # !rm ./*.gpkg
# # !rm ./*.geojson

# # # !rm ./*.tif # Double commenting this out by default so you don't have to redownload the 60MB chm.tif
