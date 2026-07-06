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
# # Shelter Indices at 1m Resolution
#
# Demonstrates running the full shelterbelts indices pipeline on a 1m canopy height model. Units are still in pixels (but now 1 pixel = 1m, compared to normally when 1 pixel = 10m).
#
# In this example, we have 4 combinations of parameters: 
#
# | # | Name | Wind Method | Notes |
# |---|------|--------|-------|
# | 1 | Default percent | percent | Tree cover within 100m is > 5% |
# | 4 | Default wind | WINDWARD | 100% shelter in the leeward direction & 50% in the windward direction |
# | 5 | More shelter | ANY | Shelter provided in any wind direction |
# | 6 | Less shelter | MOST_COMMON | Shelter only provided in the leeward direction |

# %%
from shelterbelts.indices.all_indices import indices_tif
from shelterbelts.indices.shelter_categories import shelter_categories_cmap, shelter_categories_labels
from shelterbelts.utils.visualisation import visualise_categories
from shelterbelts.utils.filepaths import _repo_root

percent_tif = str(_repo_root / 'data' / 'demo_crowns_chm_res1_500m_Sep.tif')
outdir = str(_repo_root / 'outdir')
debug = False

# %% [markdown]
# ## 1. Default percent method

# %%
# %%time
ds1, df1 = indices_tif(
    percent_tif,
    outdir=outdir,
    stub='crowns_default_percent',
    cover_threshold=1,
    crop_pixels=0,
    distance_threshold=100,
    buffer_width=40,
    max_shelterbelt_width=60,
    min_shelterbelt_length=100,
    min_patch_size=200,
    edge_size=30,
    min_core_size=10000,
    debug=debug,
)

# %%
visualise_categories(
    ds1['shelter_categories'],
    colormap=shelter_categories_cmap,
    labels=shelter_categories_labels,
    title='Percent method',
)

# %%
df1.head()

# %% [markdown]
# ## 4. Default wind method
#

# %%
# %%time
ds4, df4 = indices_tif(
    percent_tif,
    outdir=outdir,
    stub='crowns_default_wind',
    cover_threshold=1,
    crop_pixels=0,
    wind_method='WINDWARD',
    distance_threshold=100,
    buffer_width=40,
    max_shelterbelt_width=60,
    min_shelterbelt_length=100,
    min_patch_size=200,
    edge_size=30,
    min_core_size=10000,
    debug=debug,
)

# %%
visualise_categories(
    ds4['shelter_categories'],
    colormap=shelter_categories_cmap,
    labels=shelter_categories_labels,
    title='Default wind (WINDWARD)',
)

# %% [markdown]
# ## 5. More shelter — wind method
#

# %%
# %%time
ds5, df5 = indices_tif(
    percent_tif,
    outdir=outdir,
    stub='crowns_more_wind',
    cover_threshold=1,
    crop_pixels=0,
    wind_method='ANY',
    distance_threshold=100,
    buffer_width=50,
    max_shelterbelt_width=70,
    min_shelterbelt_length=100,
    min_patch_size=150,
    edge_size=50,
    min_core_size=100000,
    wind_threshold=150,
    debug=debug,
)

# %%
visualise_categories(
    ds5['shelter_categories'],
    colormap=shelter_categories_cmap,
    labels=shelter_categories_labels,
    title='More shelter windmethod (ANY)',
)

# %% [markdown]
# ## 6. Less shelter — wind method
#

# %%
# %%time
ds6, df6 = indices_tif(
    percent_tif,
    outdir=outdir,
    stub='crowns_less_wind',
    cover_threshold=1,
    crop_pixels=0,
    wind_method='MOST_COMMON',
    distance_threshold=100,
    buffer_width=30,
    max_shelterbelt_width=50,
    min_shelterbelt_length=100,
    min_patch_size=250,
    edge_size=20,
    min_core_size=1000,
    wind_threshold=25,
    debug=debug,
)

# %%
visualise_categories(
    ds6['shelter_categories'],
    colormap=shelter_categories_cmap,
    labels=shelter_categories_labels,
    title='Less shelterbelts wind (MOST_COMMON)',
)

# %% [markdown]
# ## Summary — all four side by side

# %%
import matplotlib.pyplot as plt
from shelterbelts.utils.visualisation import _plot_categories_on_axis

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
pairs = [
    (ds1, 'Default percent'),
    (ds4, 'Default wind'),
    (ds5, 'More shelter'),
    (ds6, 'Less shelter'),
]
for ax, (ds, title) in zip(axes.flat, pairs):
    _plot_categories_on_axis(ax, ds['shelter_categories'], shelter_categories_cmap, shelter_categories_labels, title, legend_inside=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Cleanup

# %%
# # !rm outdir/crowns_*.tif
# # !rm outdir/crowns_*.csv
# # !rm outdir/crowns_*.gpkg
