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
# # Model accuracy vs. distance from the treeline
#
# For each tile we load Nick's tree-cover label, each model prediction, and the
# shelter_distances raster (0 = inside trees, 1–20 = pixels outside).  We accumulate
# true-positive/false-positive/false-negative/true-negative counts per (distance, model)
# pair across all tiles, then derive precision / recall / accuracy.

# %%
import os

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
canopy_height_folder_v1  = '/scratch/xe2/cb8590/Nick_GCH'
canopy_height_folder_v2  = '/scratch/xe2/cb8590/Nick_GCH_v2'
shelter_distances_folder = '/scratch/xe2/cb8590/Nick_indices_distances_scatfix'
qld_woody_folder         = '/scratch/xe2/cb8590/Nick_Queensland_Woody'

# %% [markdown]
# ## 1. Build tile catalogue — keep only tiles that have a shelter_distances tif

# %%
gdf_nodesert = gpd.read_file(f'{nick_outlines}/footprints_large_recent_nodesert.gpkg')
gdf_nodesert['stub'] = [f.split('.')[0] for f in gdf_nodesert['filename']]

shelter_stubs = {
    f.replace('_shelter_distances.tif', '')
    for f in os.listdir(shelter_distances_folder)
    if f.endswith('_shelter_distances.tif')
}
qld_stubs = {
    f.replace('_qld_woody.tif', '')
    for f in os.listdir(qld_woody_folder)
    if f.endswith('_qld_woody.tif')
}
gdf_tiles = gdf_nodesert[
    gdf_nodesert['stub'].isin(shelter_stubs) & gdf_nodesert['stub'].isin(qld_stubs)
].reset_index(drop=True)
print(f"{len(gdf_tiles)} tiles with shelter_distances and qld_woody")

# %% [markdown]
# ## 2. Accumulate TP / FP / FN / TN per (distance, model) across all tiles

# %%
DISTANCES = list(range(0, 21))  # 0 = inside trees, 1–20 = pixels outside
MODELS = [
    'global_canopy_height_v1_trees',
    'global_canopy_height_v2_trees',
    'worldcover_trees',
    'qld_woody_trees',
    'my_predictions_50',
    'my_predictions_60',
    'my_predictions_70',
    'my_predictions_80',
    'my_predictions_90',
]


# %%
def load_binary(path, match_da, resampling=Resampling.nearest):
    """Open a tif, reproject-match to match_da, return as int DataArray."""
    da = rxr.open_rasterio(path).isel(band=0).drop_vars('band')
    return da.rio.reproject_match(match_da, resampling=resampling).astype(int)


def create_df_distances(limit=None):
    rows = []
    tiles = gdf_tiles if limit is None else gdf_tiles.iloc[:limit]

    for _, row in tiles.iterrows():
        tile = row['stub']

        label_path = f'{nick_aus_treecover_10m}/{tile}.tiff'
        pred_path  = f'{my_prediction_folder}/{tile}_my_prediction.tif'
        dist_path  = f'{shelter_distances_folder}/{tile}_shelter_distances.tif'
        wc_path    = f'{worldcover_folder}/{tile}_worldcover.tif'
        gch1_path  = f'{canopy_height_folder_v1}/{tile}_canopy_height.tif'
        gch2_path  = f'{canopy_height_folder_v2}/{tile}_canopy_height.tif'
        qld_path   = f'{qld_woody_folder}/{tile}_qld_woody.tif'

        required = [label_path, pred_path, dist_path, wc_path, gch1_path, gch2_path, qld_path]
        if not all(os.path.exists(p) for p in required):
            continue

        # Reference grid: use the prediction tif
        ds_pred   = rxr.open_rasterio(pred_path).isel(band=0).drop_vars('band')
        ds_pred_trees = (ds_pred >= 50).astype(int)

        ds_label  = load_binary(label_path, ds_pred_trees)
        ds_dist   = load_binary(dist_path,  ds_pred_trees)
        ds_wc     = rxr.open_rasterio(wc_path).isel(band=0).drop_vars('band').rio.reproject_match(ds_pred_trees, resampling=Resampling.nearest)
        ds_wc_trees  = ((ds_wc == 10) | (ds_wc == 20)).astype(int)
        ds_gch1   = load_binary(gch1_path, ds_pred_trees)
        ds_gch1_trees = (ds_gch1 >= 1).astype(int)
        ds_gch2   = load_binary(gch2_path, ds_pred_trees)
        ds_gch2_trees = (ds_gch2 >= 1).astype(int)
        ds_qld    = load_binary(qld_path, ds_pred_trees, resampling=Resampling.max)
        ds_qld_trees = (ds_qld >= 1).astype(int)

        preds = {
            'worldcover_trees':              ds_wc_trees.values,
            'global_canopy_height_v1_trees': ds_gch1_trees.values,
            'global_canopy_height_v2_trees': ds_gch2_trees.values,
            'my_predictions_50':             (ds_pred.values >= 50).astype(int),
            'my_predictions_60':             (ds_pred.values >= 60).astype(int),
            'my_predictions_70':             (ds_pred.values >= 70).astype(int),
            'my_predictions_80':             (ds_pred.values >= 80).astype(int),
            'my_predictions_90':             (ds_pred.values >= 90).astype(int),
            'qld_woody_trees':               ds_qld_trees.values,
        }

        y_true_all = ds_label.values.ravel()
        dist_all   = ds_dist.values.ravel()

        for d in DISTANCES:
            # At d>0, scattered tree pixels (category 11) can receive non-zero distances
            # because they are excluded from the shelter mask (only categories>=12 provide shelter).
            # Filter to nick_trees==0 at d>0 so we only evaluate genuine open-field pixels.
            if d == 0:
                mask = dist_all == d
            else:
                mask = (dist_all == d) & (y_true_all == 0)
            if mask.sum() == 0:
                continue
            y_true = y_true_all[mask]
            for model, pred_arr in preds.items():
                y_pred = pred_arr.ravel()[mask]
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                rows.append({'tile': tile, 'distance': d, 'model': model,
                             'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})

    df = pd.DataFrame(rows)
    return df


# %%
# Quick timing check with 10 tiles
import time
t0 = time.time()
df_sample = create_df_distances(limit=10)
elapsed = time.time() - t0
estimated_mins = elapsed / 10 * len(gdf_tiles) / 60
print(f"10 tiles in {elapsed:.1f}s → estimated full run: {estimated_mins:.1f} min")

# %%
if estimated_mins < 30:
    print("Running full dataset...")
    df_raw = create_df_distances()
    df_raw.to_csv(f'{nick_outlines}/df_distances_raw.csv', index=False)
    print(f"Saved {len(df_raw)} rows to df_distances_raw.csv")
else:
    print(f"Estimated {estimated_mins:.0f} min — too slow, using sample only")
    df_raw = df_sample

# %% [markdown]
# ## 3. Aggregate TP / FP / FN / TN → precision / recall / accuracy

# %%
agg = df_raw.groupby(['distance', 'model'])[['tp', 'fp', 'fn', 'tn']].sum().reset_index()
agg['n']          = agg['tp'] + agg['fp'] + agg['fn'] + agg['tn']
agg['accuracy']   = (agg['tp'] + agg['tn']) / agg['n']
agg['precision']  = agg['tp'] / (agg['tp'] + agg['fp']).replace(0, np.nan)
agg['recall']     = agg['tp'] / (agg['tp'] + agg['fn']).replace(0, np.nan)
agg['specificity'] = agg['tn'] / (agg['tn'] + agg['fp']).replace(0, np.nan)

agg.to_csv(f'{nick_outlines}/df_distances_summary.csv', index=False)
print("Saved df_distances_summary.csv")
agg

# %% [markdown]
# ## 4. Visualise accuracy vs. distance

# %%
model_labels = {
    'global_canopy_height_v1_trees': 'GCH v1',
    'global_canopy_height_v2_trees': 'GCH v2',
    'worldcover_trees':              'WorldCover',
    'qld_woody_trees':               'QLD Woody',
    'my_predictions_50':             'My pred (0.50)',
    'my_predictions_60':             'My pred (0.60)',
    'my_predictions_70':             'My pred (0.70)',
    'my_predictions_80':             'My pred (0.80)',
    'my_predictions_90':             'My pred (0.90)',
}
colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#99000d']

plot_distances = list(range(1, 21))

# Pixel counts at each distance (same for all models — use my_predictions)
counts = (
    df_raw[(df_raw['model'] == 'my_predictions_50') & (df_raw['distance'] >= 1)]
    .groupby('distance')[['tp', 'fp', 'fn', 'tn']].sum()
    .assign(total=lambda d: d['tp'] + d['fp'] + d['fn'] + d['tn'])
    .reindex(plot_distances)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: false positive rate (log scale) per distance per model
ax = axes[0]
for model, color in zip(MODELS, colors):
    df_m = agg[(agg['model'] == model) & (agg['distance'] >= 1)].sort_values('distance')
    fpr = np.maximum(1 - df_m['accuracy'].values, 1e-6)
    ax.plot(df_m['distance'], fpr, label=model_labels[model],
            color=color, marker='o', markersize=4)
ax.set_yscale('log')
ax.set_xlabel('Distance from treeline (pixels)')
ax.set_ylabel('False positive rate (log scale)')
ax.set_title('Model false positive rate vs. distance outside treeline\n(QLD tiles only)')
ax.set_xticks(plot_distances)
ax.set_xticklabels(plot_distances, fontsize=7)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Right: pixel count per distance
ax2 = axes[1]
ax2.bar(plot_distances, counts['total'].values / 1e6, color='steelblue', width=0.7)
ax2.set_xlabel('Distance from treeline (pixels)')
ax2.set_ylabel('Pixel count (millions)')
ax2.set_title('Non-tree pixels per distance band\n(QLD tiles only)')
ax2.set_xticks(plot_distances)
ax2.set_xticklabels(plot_distances, fontsize=7)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Model false positive rate vs. distance from the treeline (outside trees)', y=1.02)
plt.tight_layout()
plt.savefig(f'{nick_outlines}/accuracy_vs_distance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved accuracy_vs_distance.png")

# %% [markdown]
# ## 5. Pivot table: accuracy at each distance × model

# %%
pivot = agg[(agg['distance'] >= 1)].pivot(index='distance', columns='model', values='accuracy')[MODELS]
pivot.columns = [model_labels[m] for m in MODELS]
pivot.round(3)
