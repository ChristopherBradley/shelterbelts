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
# # Model accuracy vs. distance inside the treeline
#
# For each QLD tile we load Nick's tree-cover label, each model prediction, and three
# linear/tree-category rasters (edge_size 1, 2, 3).  Each tree pixel (nick_trees=1) is
# assigned to an edge band based on how far it sits from the patch boundary:
#
# | Band | Condition |
# |------|-----------|
# | Edge 1 | `lc_edge1 == 13`  (Patch Edge at edge_size=1) |
# | Edge 2 | `lc_edge1 == 12`  AND `lc_edge2 == 13`  (core at size 1, edge at size 2) |
# | Edge 3 | `lc_edge2 == 12`  AND `tc_dist == 13`   (core at size 2, edge at size 3) |
# | Core (4+) | `tc_dist == 12`  (core even at edge_size=3) |
#
# All pixels here have nick_trees=1, so precision/recall aren't meaningful — we report
# accuracy (= recall among tree pixels).

# %%
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.enums import Resampling

from shelterbelts.utils.filepaths import (
    my_prediction_folder,
    nick_aus_treecover_10m,
    nick_outlines,
    worldcover_folder,
)

# %%
canopy_height_folder_v1 = '/scratch/xe2/cb8590/Nick_GCH'
canopy_height_folder_v2 = '/scratch/xe2/cb8590/Nick_GCH_v2'
edge1_folder            = '/scratch/xe2/cb8590/Nick_indices_edge1'
edge2_folder            = '/scratch/xe2/cb8590/Nick_indices_edge2'
dist_folder             = '/scratch/xe2/cb8590/Nick_indices_distances'
qld_woody_folder        = '/scratch/xe2/cb8590/Nick_Queensland_Woody'

# %% [markdown]
# ## 1. Build tile catalogue — QLD tiles present in all three index directories

# %%
gdf_nodesert = gpd.read_file(f'{nick_outlines}/footprints_large_recent_nodesert.gpkg')
gdf_nodesert['stub'] = [f.split('.')[0] for f in gdf_nodesert['filename']]

edge1_stubs    = {f.replace('_linear_categories.tif', '') for f in os.listdir(edge1_folder) if f.endswith('_linear_categories.tif')}
edge2_stubs    = {f.replace('_linear_categories.tif', '') for f in os.listdir(edge2_folder) if f.endswith('_linear_categories.tif')}
dist_tree_stubs= {f.replace('_tree_categories.tif',  '') for f in os.listdir(dist_folder)  if f.endswith('_tree_categories.tif')}
qld_stubs      = {f.replace('_qld_woody.tif',        '') for f in os.listdir(qld_woody_folder) if f.endswith('_qld_woody.tif')}

gdf_tiles = gdf_nodesert[
    gdf_nodesert['stub'].isin(edge1_stubs) &
    gdf_nodesert['stub'].isin(edge2_stubs) &
    gdf_nodesert['stub'].isin(dist_tree_stubs) &
    gdf_nodesert['stub'].isin(qld_stubs)
].reset_index(drop=True)
print(f"{len(gdf_tiles)} tiles with edge1, edge2, dist_tree, and qld_woody")

# %% [markdown]
# ## 2. Accumulate TP / FN per (band, model) across all tiles

# %%
BANDS = ['Edge 1', 'Edge 2', 'Edge 3', 'Core (4+)']
MODELS = [
    'worldcover_trees',
    'global_canopy_height_v1_trees',
    'global_canopy_height_v2_trees',
    'my_predictions',
    'qld_woody_trees',
]


# %%
def load_categorical(path, match_da):
    """Open a categorical tif, reproject-match with nearest resampling, return as int DataArray."""
    da = rxr.open_rasterio(path).isel(band=0).drop_vars('band')
    return da.rio.reproject_match(match_da, resampling=Resampling.nearest).astype(int)


def create_df_edges(limit=None):
    rows = []
    tiles = gdf_tiles if limit is None else gdf_tiles.iloc[:limit]

    for _, row in tiles.iterrows():
        tile = row['stub']

        label_path   = f'{nick_aus_treecover_10m}/{tile}.tiff'
        pred_path    = f'{my_prediction_folder}/{tile}_my_prediction.tif'
        wc_path      = f'{worldcover_folder}/{tile}_worldcover.tif'
        gch1_path    = f'{canopy_height_folder_v1}/{tile}_canopy_height.tif'
        gch2_path    = f'{canopy_height_folder_v2}/{tile}_canopy_height.tif'
        qld_path     = f'{qld_woody_folder}/{tile}_qld_woody.tif'
        edge1_path   = f'{edge1_folder}/{tile}_linear_categories.tif'
        edge2_path   = f'{edge2_folder}/{tile}_linear_categories.tif'
        tc_dist_path = f'{dist_folder}/{tile}_tree_categories.tif'

        required = [label_path, pred_path, wc_path, gch1_path, gch2_path, qld_path,
                    edge1_path, edge2_path, tc_dist_path]
        if not all(os.path.exists(p) for p in required):
            continue

        # Reference grid: use prediction tif
        ds_pred      = rxr.open_rasterio(pred_path).isel(band=0).drop_vars('band')
        ds_pred_trees = (ds_pred >= 50).astype(int)

        ds_label = load_categorical(label_path, ds_pred_trees)

        # Edge-band assignment rasters
        lc_e1   = load_categorical(edge1_path,   ds_pred_trees)
        lc_e2   = load_categorical(edge2_path,   ds_pred_trees)
        tc_dist = load_categorical(tc_dist_path, ds_pred_trees)

        # Model predictions
        ds_wc        = rxr.open_rasterio(wc_path).isel(band=0).drop_vars('band').rio.reproject_match(ds_pred_trees, resampling=Resampling.nearest)
        ds_wc_trees  = ((ds_wc == 10) | (ds_wc == 20)).astype(int)
        ds_gch1      = load_categorical(gch1_path, ds_pred_trees)
        ds_gch1_trees = (ds_gch1 >= 1).astype(int)
        ds_gch2      = load_categorical(gch2_path, ds_pred_trees)
        ds_gch2_trees = (ds_gch2 >= 1).astype(int)
        ds_qld       = load_categorical(qld_path, ds_pred_trees)
        ds_qld_trees = (ds_qld >= 1).astype(int)

        y_true = ds_label.values.ravel()
        e1_arr  = lc_e1.values.ravel()
        e2_arr  = lc_e2.values.ravel()
        tc_arr  = tc_dist.values.ravel()

        tree_mask = y_true == 1

        band_masks = {
            'Edge 1':    tree_mask & (e1_arr == 13),
            'Edge 2':    tree_mask & (e1_arr != 13) & (e2_arr == 13),
            'Edge 3':    tree_mask & (e2_arr != 13) & (tc_arr == 13),
            'Core (4+)': tree_mask & (tc_arr == 12),
        }

        preds = {
            'worldcover_trees':              ds_wc_trees.values.ravel(),
            'global_canopy_height_v1_trees': ds_gch1_trees.values.ravel(),
            'global_canopy_height_v2_trees': ds_gch2_trees.values.ravel(),
            'my_predictions':                ds_pred_trees.values.ravel(),
            'qld_woody_trees':               ds_qld_trees.values.ravel(),
        }

        for band, mask in band_masks.items():
            if mask.sum() == 0:
                continue
            for model, pred_arr in preds.items():
                y_pred = pred_arr[mask]
                tp = int((y_pred == 1).sum())
                fn = int((y_pred == 0).sum())
                rows.append({'tile': tile, 'band': band, 'model': model, 'tp': tp, 'fn': fn})

    return pd.DataFrame(rows)


# %%
# Quick timing check with 10 tiles
t0 = time.time()
df_sample = create_df_edges(limit=10)
elapsed = time.time() - t0
estimated_mins = elapsed / 10 * len(gdf_tiles) / 60
print(f"10 tiles in {elapsed:.1f}s → estimated full run: {estimated_mins:.1f} min")

# %%
if estimated_mins < 30:
    print("Running full dataset...")
    df_raw = create_df_edges()
    df_raw.to_csv(f'{nick_outlines}/df_edges_raw.csv', index=False)
    print(f"Saved {len(df_raw)} rows to df_edges_raw.csv")
else:
    print(f"Estimated {estimated_mins:.0f} min — too slow, using sample only")
    df_raw = df_sample

# %% [markdown]
# ## 3. Aggregate TP / FN → accuracy

# %%
agg = df_raw.groupby(['band', 'model'])[['tp', 'fn']].sum().reset_index()
agg['n']        = agg['tp'] + agg['fn']
agg['accuracy'] = agg['tp'] / agg['n']  # == recall since every pixel has nick_trees=1

agg.to_csv(f'{nick_outlines}/df_edges_summary.csv', index=False)
print("Saved df_edges_summary.csv")
agg

# %% [markdown]
# ## 4. Pixel counts per band

# %%
# Use a single model's counts (mask is the same regardless of model)
counts = (
    df_raw[df_raw['model'] == 'my_predictions']
    .groupby('band')[['tp', 'fn']].sum()
    .assign(total=lambda d: d['tp'] + d['fn'])
    .reindex(BANDS)
)
counts['pct'] = counts['total'] / counts['total'].sum() * 100
counts[['total', 'pct']].round(1)

# %% [markdown]
# ## 5. Visualise

# %%
model_labels = {
    'worldcover_trees':              'WorldCover',
    'global_canopy_height_v1_trees': 'GCH v1',
    'global_canopy_height_v2_trees': 'GCH v2',
    'my_predictions':                'My predictions',
    'qld_woody_trees':               'QLD Woody',
}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: accuracy per band per model (line chart)
ax = axes[0]
x = np.arange(len(BANDS))
for model, color in zip(MODELS, colors):
    df_m = agg[agg['model'] == model].set_index('band').reindex(BANDS)
    ax.plot(x, df_m['accuracy'].values, label=model_labels[model],
            color=color, marker='o', markersize=6)
ax.set_xticks(x)
ax.set_xticklabels(BANDS)
ax.set_ylabel('Accuracy (= recall among tree pixels)')
ax.set_title('Model accuracy vs. distance inside treeline\n(QLD tiles only)')
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Right: pixel count per band
ax2 = axes[1]
bar_colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb']
bars = ax2.bar(BANDS, counts['total'].values / 1e6, color=bar_colors, edgecolor='white')
for bar, pct in zip(bars, counts['pct'].values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)
ax2.set_ylabel('Pixel count (millions)')
ax2.set_title('Tree pixels per band\n(% of classified tree pixels)')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Model accuracy inside the treeline — distance from patch boundary', y=1.02)
plt.tight_layout()
plt.savefig(f'{nick_outlines}/accuracy_vs_edge.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved accuracy_vs_edge.png")

# %% [markdown]
# ## 6. Summary table: accuracy per band × model

# %%
pivot = agg.pivot(index='band', columns='model', values='accuracy')[MODELS].reindex(BANDS)
pivot.columns = [model_labels[m] for m in MODELS]
pivot.round(3)
