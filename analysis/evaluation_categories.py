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
# # Model accuracy per tree category
#
# For each QLD tile we load Nick's tree-cover label, each model prediction, and the
# linear_categories raster from Nick_indices_distances.  For each tree pixel (nick_trees=1)
# we assign it to the category reported by linear_categories (10–19) and accumulate
# TP / FN per (category, model).
#
# Since every pixel in the analysis has nick_trees=1, accuracy == recall among tree pixels.
#
# | Category | Label |
# |----------|-------|
# | 10 | Tree cover |
# | 11 | Scattered Trees |
# | 12 | Patch Core |
# | 13 | Patch Edge |
# | 14 | Corridor (Other Trees) |
# | 15 | Gully Trees |
# | 16 | Ridge Trees |
# | 17 | Road Trees |
# | 18 | Linear Patches |
# | 19 | Non-linear Patches |

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
dist_folder             = '/scratch/xe2/cb8590/Nick_indices_distances'
qld_woody_folder        = '/scratch/xe2/cb8590/Nick_Queensland_Woody'

# %% [markdown]
# ## 1. Build tile catalogue — QLD tiles with linear_categories in dist folder

# %%
gdf_nodesert = gpd.read_file(f'{nick_outlines}/footprints_large_recent_nodesert.gpkg')
gdf_nodesert['stub'] = [f.split('.')[0] for f in gdf_nodesert['filename']]

dist_lc_stubs = {f.replace('_linear_categories.tif', '') for f in os.listdir(dist_folder) if f.endswith('_linear_categories.tif')}
qld_stubs     = {f.replace('_qld_woody.tif', '')         for f in os.listdir(qld_woody_folder) if f.endswith('_qld_woody.tif')}

gdf_tiles = gdf_nodesert[
    gdf_nodesert['stub'].isin(dist_lc_stubs) &
    gdf_nodesert['stub'].isin(qld_stubs)
].reset_index(drop=True)
print(f"{len(gdf_tiles)} tiles with linear_categories and qld_woody")

# %% [markdown]
# ## 2. Accumulate TP / FN per (category, model) across all tiles

# %%
CATEGORIES = [11, 12, 13, 15, 17, 18, 19]
CATEGORY_LABELS = {
    11: 'Scattered Trees',
    12: 'Patch Core',
    13: 'Patch Edge',
    15: 'Trees in Gullies',
    17: 'Trees next to Roads',
    18: 'Linear Patches',
    19: 'Non-linear Patches',
}
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
def load_categorical(path, match_da):
    """Open a categorical tif, reproject-match with nearest resampling, return as int DataArray."""
    da = rxr.open_rasterio(path).isel(band=0).drop_vars('band')
    return da.rio.reproject_match(match_da, resampling=Resampling.nearest).astype(int)


def create_df_categories(limit=None):
    rows = []
    tiles = gdf_tiles if limit is None else gdf_tiles.iloc[:limit]

    for _, row in tiles.iterrows():
        tile = row['stub']

        label_path = f'{nick_aus_treecover_10m}/{tile}.tiff'
        pred_path  = f'{my_prediction_folder}/{tile}_my_prediction.tif'
        wc_path    = f'{worldcover_folder}/{tile}_worldcover.tif'
        gch1_path  = f'{canopy_height_folder_v1}/{tile}_canopy_height.tif'
        gch2_path  = f'{canopy_height_folder_v2}/{tile}_canopy_height.tif'
        qld_path   = f'{qld_woody_folder}/{tile}_qld_woody.tif'
        lc_path    = f'{dist_folder}/{tile}_linear_categories.tif'

        required = [label_path, pred_path, wc_path, gch1_path, gch2_path, qld_path, lc_path]
        if not all(os.path.exists(p) for p in required):
            continue

        ds_pred      = rxr.open_rasterio(pred_path).isel(band=0).drop_vars('band')
        ds_pred_trees = (ds_pred >= 50).astype(int)

        ds_label = load_categorical(label_path, ds_pred_trees)
        lc       = load_categorical(lc_path, ds_pred_trees)

        ds_wc       = rxr.open_rasterio(wc_path).isel(band=0).drop_vars('band').rio.reproject_match(ds_pred_trees, resampling=Resampling.nearest)
        ds_wc_trees  = ((ds_wc == 10) | (ds_wc == 20)).astype(int)
        ds_gch1     = load_categorical(gch1_path, ds_pred_trees)
        ds_gch1_trees = (ds_gch1 >= 1).astype(int)
        ds_gch2     = load_categorical(gch2_path, ds_pred_trees)
        ds_gch2_trees = (ds_gch2 >= 1).astype(int)
        ds_qld      = load_categorical(qld_path, ds_pred_trees)
        ds_qld_trees = (ds_qld >= 1).astype(int)

        y_true  = ds_label.values.ravel()
        lc_arr  = lc.values.ravel()

        preds = {
            'worldcover_trees':              ds_wc_trees.values.ravel(),
            'global_canopy_height_v1_trees': ds_gch1_trees.values.ravel(),
            'global_canopy_height_v2_trees': ds_gch2_trees.values.ravel(),
            'my_predictions_50':             (ds_pred.values.ravel() >= 50).astype(int),
            'my_predictions_60':             (ds_pred.values.ravel() >= 60).astype(int),
            'my_predictions_70':             (ds_pred.values.ravel() >= 70).astype(int),
            'my_predictions_80':             (ds_pred.values.ravel() >= 80).astype(int),
            'my_predictions_90':             (ds_pred.values.ravel() >= 90).astype(int),
            'qld_woody_trees':               ds_qld_trees.values.ravel(),
        }

        for cat in CATEGORIES:
            # Tree pixels (nick_trees=1) assigned to this linear category
            mask = (y_true == 1) & (lc_arr == cat)
            if mask.sum() == 0:
                continue
            for model, pred_arr in preds.items():
                y_pred = pred_arr[mask]
                tp = int((y_pred == 1).sum())
                fn = int((y_pred == 0).sum())
                rows.append({'tile': tile, 'category': cat, 'model': model, 'tp': tp, 'fn': fn})

    return pd.DataFrame(rows)


# %%
# Quick timing check with 10 tiles
t0 = time.time()
df_sample = create_df_categories(limit=10)
elapsed = time.time() - t0
estimated_mins = elapsed / 10 * len(gdf_tiles) / 60
print(f"10 tiles in {elapsed:.1f}s → estimated full run: {estimated_mins:.1f} min")

# %%
if estimated_mins < 30:
    print("Running full dataset...")
    df_raw = create_df_categories()
    df_raw.to_csv(f'{nick_outlines}/df_categories_raw.csv', index=False)
    print(f"Saved {len(df_raw)} rows to df_categories_raw.csv")
else:
    print(f"Estimated {estimated_mins:.0f} min — too slow, using sample only")
    df_raw = df_sample

# %% [markdown]
# ## 3. Aggregate TP / FN → accuracy

# %%
agg = df_raw.groupby(['category', 'model'])[['tp', 'fn']].sum().reset_index()
agg['n']        = agg['tp'] + agg['fn']
agg['accuracy'] = agg['tp'] / agg['n']  # == recall since every pixel has nick_trees=1

agg.to_csv(f'{nick_outlines}/df_categories_summary.csv', index=False)
print("Saved df_categories_summary.csv")
agg

# %% [markdown]
# ## 4. Pixel counts per category

# %%
counts = (
    df_raw[df_raw['model'] == 'my_predictions_50']
    .groupby('category')[['tp', 'fn']].sum()
    .assign(total=lambda d: d['tp'] + d['fn'])
    .reindex(CATEGORIES)
    .fillna(0)
)
counts['pct'] = counts['total'] / counts['total'].sum() * 100
counts[['total', 'pct']].round(1)

# %% [markdown]
# ## 5. Visualise

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
cat_tick_labels = [CATEGORY_LABELS[c] for c in CATEGORIES]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: accuracy per category per model (grouped bar chart)
ax = axes[0]
x = np.arange(len(CATEGORIES))
n_models = len(MODELS)
width = 0.8 / n_models
for i, (model, color) in enumerate(zip(MODELS, colors)):
    df_m = agg[agg['model'] == model].set_index('category').reindex(CATEGORIES)
    offset = (i - n_models / 2 + 0.5) * width
    ax.bar(x + offset, df_m['accuracy'].values, width=width,
           label=model_labels[model], color=color)
ax.set_xticks(x)
ax.set_xticklabels(cat_tick_labels, rotation=30, ha='right')
ax.set_ylabel('Accuracy (= recall among tree pixels)')
ax.set_title('Model accuracy per tree category\n(QLD tiles only)')
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Right: pixel count per category
ax2 = axes[1]
bars = ax2.bar(x, counts['total'].values / 1e6, color='steelblue', width=0.7)
for bar, pct in zip(bars, counts['pct'].values):
    if pct >= 1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(cat_tick_labels, rotation=30, ha='right')
ax2.set_ylabel('Pixel count (millions)')
ax2.set_title('Tree pixels per category\n(% of total tree pixels)')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Model accuracy per tree category', y=1.02)
plt.tight_layout()
plt.savefig(f'{nick_outlines}/accuracy_vs_category.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved accuracy_vs_category.png")

# %% [markdown]
# ## 6. Summary table: accuracy per category × model

# %%
pivot = agg.pivot(index='category', columns='model', values='accuracy')[MODELS].reindex(CATEGORIES)
pivot.index = [CATEGORY_LABELS[c] for c in CATEGORIES]
pivot.columns = [model_labels[m] for m in MODELS]
pivot.round(3)
