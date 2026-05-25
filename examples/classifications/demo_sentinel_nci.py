# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Download Sentinel-2 via the NCI Datacube
#
# Fetch Sentinel-2 surface-reflectance imagery directly from the
# [NCI](https://nci.org.au/) datacube using ``datacube`` and ``dea_tools``.
#
# **Requirements**: must be run inside the DEA/NCI sandbox or on Gadi with
# the ``shelterbelts`` conda environment loaded.

# %%
import pickle

import datacube
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr

from shelterbelts.classifications.sentinel_nci import define_query_range, load_and_process_data, download_ds2
from shelterbelts.utils.filepaths import get_filename

# %% [markdown]
# ## Configure the area and date range
#
# We derive the bounding box from the binary-tree-cover tif and convert to
# EPSG:4326 lat/lon ranges for the datacube query.

# %%
tif = get_filename('g2_26729_binary_tree_cover_10m.tiff')
da = rxr.open_rasterio(tif).isel(band=0).drop_vars('band')
bbox = da.rio.reproject('EPSG:4326').rio.bounds()
print(f"Bounding box (EPSG:4326): {bbox}")

start_date = "2020-01-01"
end_date   = "2020-04-01"

lat_range = (bbox[1], bbox[3])
lon_range = (bbox[0], bbox[2])

# %% [markdown]
# ## Build the datacube query

# %%
query = define_query_range(lat_range, lon_range, time_range=(start_date, end_date))
query

# %% [markdown]
# ## Load the imagery
#
# ``load_and_process_data`` calls ``load_ard`` with a 90 % good-data threshold
# and pixel-level cloud masking via ``s2cloudless``.

# %%
# %%time
dc = datacube.Datacube(app='demo_sentinel_nci')
ds = load_and_process_data(dc, query)
ds

# %% [markdown]
# ## Example Visualisation

# %%
date_idx = 0
date_str = str(ds.time.values[date_idx])[:10]

red   = ds['nbart_red'].isel(time=date_idx).values.astype(float)
green = ds['nbart_green'].isel(time=date_idx).values.astype(float)
blue  = ds['nbart_blue'].isel(time=date_idx).values.astype(float)

# Stack and normalise to [0, 1] using a simple percentile stretch
rgb = np.stack([red, green, blue], axis=-1)
lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(rgb)
ax.set_title(f"Sentinel-2 RGB  —  {date_str}")
ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save to disk
#
# ``download_ds2`` wraps the query, load, and pickle steps above.

# %%
# %%time
ds = download_ds2(tif, start_date=start_date, end_date=end_date, outdir='.')
