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
# # Shelter Categories Demo
#
# Demonstrates the `shelter_categories` function with different parameter values.

# %% [markdown]
# ## Setup

# %%
from shelterbelts.utils.filepaths import get_filename as get_example_data, get_example_tree_categories_data
from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.shelter_categories import shelter_categories_cmap, shelter_categories_labels

# Load test data
ds_cat = get_example_tree_categories_data()
wind_file = get_example_data('g2_26729_barra_daily.nc')
height_file = get_example_data('g2_26729_canopy_height.tif')

# %% [markdown]
# ## Default Parameters
#
# First, let's run with default parameters:

# %%
ds_default = shelter_categories(ds_cat, outdir='/tmp', stub='default', plot=False, savetif=False)
visualise_categories(
    ds_default['shelter_categories'],
    colormap=shelter_categories_cmap,
    labels=shelter_categories_labels
)

# %% [markdown]
# ## Parameter: density_threshold
#
# The `density_threshold` parameter sets the minimum percentage tree cover (within a radius) that counts as sheltered.
#
# - **Low value (3%)**: More areas considered sheltered
# - **High value (10%)**: Fewer areas considered sheltered, requires denser tree cover

# %%
ds1 = shelter_categories(ds_cat, outdir='/tmp', stub='dens1', plot=False, savetif=False, density_threshold=3)
ds2 = shelter_categories(ds_cat, outdir='/tmp', stub='dens2', plot=False, savetif=False, density_threshold=10)
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="density_threshold=3%", title2="density_threshold=10%"
)

# %% [markdown]
# ## Parameter: wind_method
#
# The `wind_method` parameter determines how wind direction(s) are considered:
#
# - **MOST_COMMON**: Only downwind shelter using the most common wind direction
# - **WINDWARD**: Both downwind (full distance) and upwind (half distance) shelter
# - **HAPPENED**: Shelter from any direction where winds exceeded the threshold
# - **ANY**: Shelter from all 8 compass directions regardless of wind

# %%
ds1 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wind1', plot=False, savetif=False, wind_method='MOST_COMMON')
ds2 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wind2', plot=False, savetif=False, wind_method='WINDWARD')
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="wind_method=MOST_COMMON", title2="wind_method=WINDWARD"
)

# %%
ds1 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wind3', plot=False, savetif=False, wind_method='HAPPENED', wind_threshold=28)
ds2 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wind4', plot=False, savetif=False, wind_method='ANY')
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="wind_method=HAPPENED", title2="wind_method=ANY"
)

# %% [markdown]
# ## Parameter: distance_threshold
#
# The `distance_threshold` parameter defines how far from trees a pixel is still considered sheltered.
# Units are either pixels (without height data) or tree heights (with height data).
#
# - **Low value (10)**: Tight shelter zone
# - **High value (30)**: Extended shelter zone

# %%
ds1 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='dist1', plot=False, savetif=False, distance_threshold=10)
ds2 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='dist2', plot=False, savetif=False, distance_threshold=30)
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="distance_threshold=10", title2="distance_threshold=30"
)



# %% [markdown]
# ## Parameter: wind_threshold
#
# The `wind_threshold` parameter sets the wind speed (km/h) used to determine dominant wind direction.
# Only used when `wind_data` is provided.
#
# - **Low value (10 km/h)**: Weak but regular wind can influence plant growth habits
# - **High value (30 km/h)**: Strong winds are most likely to cause major crop damage

# %%
ds1 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wt1', plot=False, savetif=False, wind_threshold=10)
ds2 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wt2', plot=False, savetif=False, wind_threshold=30)
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="wind_threshold=10 km/h", title2="wind_threshold=30 km/h"
)

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how each parameter affects shelter categorization:
# - `density_threshold`: Minimum tree cover percentage for density-based sheltering
# - `distance_threshold`: Maximum distance from trees for shelter protection
# - `wind_method`: How wind direction(s) influence shelter calculation
# - `wind_threshold`: Wind speed threshold for determining the dominant wind direction

# %%
