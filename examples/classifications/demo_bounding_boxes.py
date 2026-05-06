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
# # Bounding Boxes
#
# ``bounding_boxes`` summarises a folder of GeoTIFF tiles into a single
# GeoPackage of polygon footprints. You can open these geopackages in qgis to visualise these locations.

# %%
from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.utils.filepaths import _repo_root

# %%
gdf = bounding_boxes(
    str(_repo_root / 'data' / 'multiple_binary_tifs'),
    filetype='.tiff',
    tif_cover_threshold=5
)
gdf

# %% [markdown]
# ## Changing the size_threshold
#
# Any tif where the height or width doesn't meet this threshold is marked as 'bad_tif'. The percent_trees isn't calculated this time because we didn't provide the 'tif_cover_threshold' parameter (to save computation time).

# %%
gdf = bounding_boxes(
    str(_repo_root / 'data' / 'multiple_binary_tifs'),
    filetype='.tiff',
    size_threshold=300,
)
gdf

# %% [markdown]
# ## Changing the cover_threshold
#
# Any tif where the percentage of zeros or ones doesn't meet this threshold is marked as 'bad_tif'.

# %%
gdf = bounding_boxes(
    str(_repo_root / 'data' / 'multiple_binary_tifs'),
    filetype='.tiff',
    tif_cover_threshold=20
)
gdf

# %% [markdown]
# ## Saving Centroids
# Include a centroid GeoPackage. This is useful for visualising the spread of small tifs in very zoomed out areas.

# %%
gdf = bounding_boxes(
    str(_repo_root / 'data' / 'multiple_binary_tifs'),
    filetype='.tiff',
    save_centroids=True
)
