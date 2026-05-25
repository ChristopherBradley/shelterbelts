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
# # Download Sentinel-2 via DEA STAC (if you can't access NCI)
#
# Fetch Sentinel-2 surface-reflectance imagery from the
# public [Digital Earth Australia](https://www.dea.ga.gov.au/) STAC catalogue.
#
# **Requirements**: ``pip install "dask[distributed]" odc-stac pystac-client``
#
# * Note: I've found the DEA STAC endpoint to be a bit slow and reliable, so I've been using sentinel_nci.py for most of my work instead. This sentinel_dea.py file is mainly for demonstration purposes.

# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr

from shelterbelts.classifications.sentinel_dea import download_ds2, search_stac, load_and_process_data
from shelterbelts.utils.filepaths import get_filename

# %% [markdown]
# ## Configure the area and date range
#
# This example uses a 1km x 1km tif bounds at Milgadara. The STAC API requires
# EPSG:4326 coordinates, so we reproject the tif bounds before searching.

# %%
tif = get_filename('g2_26729_binary_tree_cover_10m.tiff')
da = rxr.open_rasterio(tif).isel(band=0).drop_vars('band')
bbox = da.rio.reproject('EPSG:4326').rio.bounds()
print(f"Bounding box (EPSG:4326): {bbox}")

start_date = "2020-01-01"
end_date   = "2020-04-01"

# %% [markdown]
# ## Search the DEA STAC catalogue

# %%
# %%time
items = search_stac(bbox, start_date, end_date)
print(f"Found {len(items)} scenes")
for item in items:
    print(f"  {item.datetime.date()}  cloud={item.properties.get('eo:cloud_cover', '?'):.1f}%")

# %% [markdown]
# ## Load the imagery
#
# This uses lazy loading so data is not downloaded until ``.compute()`` is called.

# %%
ds = load_and_process_data(items, bbox)
ds

# %% [markdown]
# ## Example Visualisation
#

# %%
# %%time
date_idx = 0
date_str = str(ds.time.values[date_idx])[:10]

red   = ds['nbart_red'].isel(time=date_idx).compute().values.astype(float)
green = ds['nbart_green'].isel(time=date_idx).compute().values.astype(float)
blue  = ds['nbart_blue'].isel(time=date_idx).compute().values.astype(float)

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
# ``download_ds2`` wraps the search + load steps above and pickles the result
# so the same file can be opened later without re-downloading.

# %%
# %%time
ds = download_ds2(tif, start_date=start_date, end_date=end_date)
