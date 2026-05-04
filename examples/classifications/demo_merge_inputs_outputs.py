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
# # Merging Sentinel inputs with binary tree labels
#
# ``merge_inputs_outputs`` combines a Sentinel-2 pickle with a binary tree-cover
# tif to produce a per-pixel training CSV. Each row is one sampled pixel, and
# columns are temporally-aggregated Sentinel features (median, std, focal
# mean, focal std per band) plus vegetation indices and the binary tree label.

# %%
import pickle
import rioxarray as rxr

from shelterbelts.classifications.merge_inputs_outputs import merge_inputs_outputs, jittered_grid, aggregated_metrics, visualise_jittered_grid
from shelterbelts.utils.filepaths import get_filename

# %% [markdown]
# ## Example Inputs

# %%
sentinel_pickle = get_filename('g2_019_sentinel_150mx150m.pkl')
tree_tif = get_filename('g2_019_binary_trees_150mx150m.tiff')

with open(sentinel_pickle, 'rb') as f:
    ds = pickle.load(f)
ds

# %%
# Example imagery (red band)
ds['nbart_red'].isel(time=0).plot(cmap='Reds_r')

# %%
# Ground truth binary tree raster
da = rxr.open_rasterio(tree_tif).isel(band=0).drop_vars('band')
da.plot(cmap='Greens')

# %% [markdown]
# ## Merge into a training CSV

# %%
df = merge_inputs_outputs(
    sentinel_pickle,
    tree_tif,
    outdir='outdir',
    spacing=3,   # default spacing=10 is designed for 1km tiles; use 3 for this 150m demo tile
)
df

# %% [markdown]
# ## Visualise the sampling grid
#
# ``merge_inputs_outputs`` uses the function ``jittered_grid`` to sample pixels at regular intervals with small random
# offsets. We can visualise this sampling by calling jittered_grid directly.

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open(sentinel_pickle, 'rb') as f:
    ds = pickle.load(f)

da = rxr.open_rasterio(tree_tif).isel(band=0).drop_vars('band')
ds = ds.rio.reproject_match(da)
ds['tree_cover'] = da.astype(float)

visualise_jittered_grid(ds, spacing=3, stub='g2_019_150mx150m')  # Can view this output in QGIS

# %%
# Example plot
grid_df = jittered_grid(ds, spacing=3)

fig, ax = plt.subplots(figsize=(5, 5))
da.plot(ax=ax, cmap='Greens', add_colorbar=False)

colors = {0.0: '#e06c3a', 1.0: '#2e7d32'}
for label, group in grid_df.groupby('tree_cover'):
    ax.scatter(group['x'], group['y'], color=colors[label], s=40, zorder=3)

patches = [
    mpatches.Patch(color=colors[0.0], label='Non-tree'),
    mpatches.Patch(color=colors[1.0], label='Tree'),
]
ax.legend(handles=patches, loc='upper right', fontsize=9)
ax.set_title('Jittered sampling grid (spacing=3)')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.show()
