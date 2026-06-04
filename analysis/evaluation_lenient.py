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
# # Lenient evaluation — categories, edges, and distances
#
# - **accuracy_vs_category** (lenient): each model's binary prediction is dilated
#   by one pixel (3×3 footprint) before computing recall. Category labels come
#   from Nick_indices_edge1 so that Patch Core / Patch Edge reflect just the
#   outermost pixel boundary of each patch.
# - **accuracy_vs_edge** (strict): standard pixel-by-pixel recall, no dilation.
# - **accuracy_vs_distance** (strict): standard pixel-by-pixel accuracy, no
#   dilation or erosion.
#
# All three plots are produced in a single pass over the tiles.

# %%
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation

from shelterbelts.utils.filepaths import (
    my_prediction_folder,
    nick_aus_treecover_10m,
    nick_outlines,
    worldcover_folder,
)

# %%
canopy_height_folder_v1  = '/scratch/xe2/cb8590/Nick_GCH'
canopy_height_folder_v2  = '/scratch/xe2/cb8590/Nick_GCH_v2'
edge1_folder             = '/scratch/xe2/cb8590/Nick_indices_edge1_scatfix'
edge2_folder             = '/scratch/xe2/cb8590/Nick_indices_edge2_scatfix'
shelter_distances_folder, dist_folder = '/scratch/xe2/cb8590/Nick_indices_distances_scatfix'
qld_woody_folder         = '/scratch/xe2/cb8590/Nick_Queensland_Woody'

FOOTPRINT = np.ones((3, 3), dtype=bool)

# %% [markdown]
# ## 1. Build tile catalogue — tiles with all required files

# %%
gdf_nodesert = gpd.read_file(f'{nick_outlines}/footprints_large_recent_nodesert.gpkg')
gdf_nodesert['stub'] = [f.split('.')[0] for f in gdf_nodesert['filename']]

tc_stubs      = {f.replace('_tree_categories.tif',   '') for f in os.listdir(dist_folder)              if f.endswith('_tree_categories.tif')}
edge1_stubs   = {f.replace('_linear_categories.tif', '') for f in os.listdir(edge1_folder)             if f.endswith('_linear_categories.tif')}
edge2_stubs   = {f.replace('_linear_categories.tif', '') for f in os.listdir(edge2_folder)             if f.endswith('_linear_categories.tif')}
shelter_stubs = {f.replace('_shelter_distances.tif', '') for f in os.listdir(shelter_distances_folder) if f.endswith('_shelter_distances.tif')}
qld_stubs     = {f.replace('_qld_woody.tif',         '') for f in os.listdir(qld_woody_folder)         if f.endswith('_qld_woody.tif')}

gdf_tiles = gdf_nodesert[
    gdf_nodesert['stub'].isin(tc_stubs) &
    gdf_nodesert['stub'].isin(edge1_stubs) &
    gdf_nodesert['stub'].isin(edge2_stubs) &
    gdf_nodesert['stub'].isin(shelter_stubs) &
    gdf_nodesert['stub'].isin(qld_stubs)
].reset_index(drop=True)
print(f"{len(gdf_tiles)} tiles with all required files")

# %% [markdown]
# ## 2. Shared constants

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

BANDS     = ['Edge 1', 'Edge 2', 'Edge 3', 'Core (4+)']
DISTANCES = list(range(0, 21))

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

# %% [markdown]
# ## 3. Accumulate stats in a single pass over tiles

# %%
def load_da(path, match_da):
    da = rxr.open_rasterio(path).isel(band=0).drop_vars('band')
    return da.rio.reproject_match(match_da, resampling=Resampling.nearest).astype(int)


def process_tiles(limit=None):
    cat_rows  = []
    edge_rows = []
    dist_rows = []
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
        shelter_path = f'{shelter_distances_folder}/{tile}_shelter_distances.tif'

        required = [label_path, pred_path, wc_path, gch1_path, gch2_path, qld_path,
                    edge1_path, edge2_path, tc_dist_path, shelter_path]
        if not all(os.path.exists(p) for p in required):
            continue

        ds_pred = rxr.open_rasterio(pred_path).isel(band=0).drop_vars('band')

        ds_label = load_da(label_path,   ds_pred)
        ds_wc    = rxr.open_rasterio(wc_path).isel(band=0).drop_vars('band').rio.reproject_match(ds_pred, resampling=Resampling.nearest)
        ds_gch1  = load_da(gch1_path,    ds_pred)
        ds_gch2  = load_da(gch2_path,    ds_pred)
        ds_qld   = load_da(qld_path,     ds_pred)
        lc_e1    = load_da(edge1_path,   ds_pred)
        lc_e2    = load_da(edge2_path,   ds_pred)
        tc_dist  = load_da(tc_dist_path, ds_pred)
        ds_dist  = load_da(shelter_path, ds_pred)

        pred_vals = ds_pred.values

        # Raw binary predictions (2D arrays)
        raw = {
            'global_canopy_height_v1_trees': (ds_gch1.values >= 1).astype(int),
            'global_canopy_height_v2_trees': (ds_gch2.values >= 1).astype(int),
            'worldcover_trees':              ((ds_wc.values == 10) | (ds_wc.values == 20)).astype(int),
            'qld_woody_trees':               (ds_qld.values >= 1).astype(int),
            'my_predictions_50':             (pred_vals >= 50).astype(int),
            'my_predictions_60':             (pred_vals >= 60).astype(int),
            'my_predictions_70':             (pred_vals >= 70).astype(int),
            'my_predictions_80':             (pred_vals >= 80).astype(int),
            'my_predictions_90':             (pred_vals >= 90).astype(int),
        }

        # Dilated predictions for lenient category recall only
        dilated = {m: binary_dilation(v.astype(bool), structure=FOOTPRINT).astype(int) for m, v in raw.items()}

        y_true   = ds_label.values.ravel()
        e1_arr   = lc_e1.values.ravel()
        e2_arr   = lc_e2.values.ravel()
        tc_arr   = tc_dist.values.ravel()
        dist_arr = ds_dist.values.ravel()
        tree_mask = y_true == 1

        # --- Categories (lenient: dilated predictions, edge1 category labels) ---
        for cat in CATEGORIES:
            mask = tree_mask & (e1_arr == cat)
            if mask.sum() == 0:
                continue
            for model in MODELS:
                y_pred = dilated[model].ravel()[mask]
                cat_rows.append({'tile': tile, 'category': cat, 'model': model,
                                 'tp': int((y_pred == 1).sum()),
                                 'fn': int((y_pred == 0).sum())})

        # --- Edges (strict: raw predictions) ---
        band_masks = {
            'Edge 1':    tree_mask & (e1_arr == 13),
            'Edge 2':    tree_mask & (e1_arr != 13) & (e2_arr == 13),
            'Edge 3':    tree_mask & (e2_arr != 13) & (tc_arr == 13),
            'Core (4+)': tree_mask & (tc_arr == 12),
        }
        for band, mask in band_masks.items():
            if mask.sum() == 0:
                continue
            for model in MODELS:
                y_pred = raw[model].ravel()[mask]
                edge_rows.append({'tile': tile, 'band': band, 'model': model,
                                  'tp': int((y_pred == 1).sum()),
                                  'fn': int((y_pred == 0).sum())})

        # --- Distances (strict: raw predictions) ---
        for d in DISTANCES:
            if d == 0:
                mask = dist_arr == d
            else:
                mask = (dist_arr == d) & (y_true == 0)
            if mask.sum() == 0:
                continue
            y_true_d = y_true[mask]
            for model in MODELS:
                y_pred = raw[model].ravel()[mask]
                dist_rows.append({'tile': tile, 'distance': d, 'model': model,
                                  'tp': int(((y_true_d == 1) & (y_pred == 1)).sum()),
                                  'fp': int(((y_true_d == 0) & (y_pred == 1)).sum()),
                                  'fn': int(((y_true_d == 1) & (y_pred == 0)).sum()),
                                  'tn': int(((y_true_d == 0) & (y_pred == 0)).sum())})

    return pd.DataFrame(cat_rows), pd.DataFrame(edge_rows), pd.DataFrame(dist_rows)


# %%
t0 = time.time()
df_cat_s, df_edge_s, df_dist_s = process_tiles(limit=10)
elapsed = time.time() - t0
estimated_mins = elapsed / 10 * len(gdf_tiles) / 60
print(f"10 tiles in {elapsed:.1f}s → estimated full run: {estimated_mins:.1f} min")

# %%
if estimated_mins < 30:
    print("Running full dataset...")
    df_cat, df_edge, df_dist = process_tiles()
    df_cat.to_csv( f'{nick_outlines}/df_categories_lenient_raw.csv', index=False)
    df_edge.to_csv(f'{nick_outlines}/df_edges_lenient_raw.csv',      index=False)
    df_dist.to_csv(f'{nick_outlines}/df_distances_lenient_raw.csv',  index=False)
    print(f"Saved {len(df_cat)} category, {len(df_edge)} edge, {len(df_dist)} distance rows")
else:
    print(f"Estimated {estimated_mins:.0f} min — too slow, using sample only")
    df_cat, df_edge, df_dist = df_cat_s, df_edge_s, df_dist_s

# %% [markdown]
# ## 4. Plot — categories

# %%
agg_cat = df_cat.groupby(['category', 'model'])[['tp', 'fn']].sum().reset_index()
agg_cat['n']        = agg_cat['tp'] + agg_cat['fn']
agg_cat['accuracy'] = agg_cat['tp'] / agg_cat['n']
agg_cat.to_csv(f'{nick_outlines}/df_categories_lenient_summary.csv', index=False)

counts_cat = (
    df_cat[df_cat['model'] == 'my_predictions_50']
    .groupby('category')[['tp', 'fn']].sum()
    .assign(total=lambda d: d['tp'] + d['fn'])
    .reindex(CATEGORIES).fillna(0)
)
counts_cat['pct'] = counts_cat['total'] / counts_cat['total'].sum() * 100
cat_tick_labels = [CATEGORY_LABELS[c] for c in CATEGORIES]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
x = np.arange(len(CATEGORIES))
n_models = len(MODELS)
width = 0.8 / n_models
for i, (model, color) in enumerate(zip(MODELS, colors)):
    df_m = agg_cat[agg_cat['model'] == model].set_index('category').reindex(CATEGORIES)
    offset = (i - n_models / 2 + 0.5) * width
    ax.bar(x + offset, df_m['accuracy'].values, width=width,
           label=model_labels[model], color=color)
ax.set_xticks(x)
ax.set_xticklabels(cat_tick_labels, rotation=30, ha='right')
ax.set_ylabel('Lenient accuracy (= recall among tree pixels)')
ax.set_title('Lenient accuracy per tree category\n(QLD tiles only)')
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
bars = ax2.bar(x, counts_cat['total'].values / 1e6, color='steelblue', width=0.7)
for bar, pct in zip(bars, counts_cat['pct'].values):
    if pct >= 1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(cat_tick_labels, rotation=30, ha='right')
ax2.set_ylabel('Pixel count (millions, log scale)')
ax2.set_title('Tree pixels per category\n(% of total tree pixels)')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Lenient model accuracy per tree category', y=1.02)
plt.tight_layout()
plt.savefig(f'{nick_outlines}/accuracy_vs_category_lenient.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved accuracy_vs_category_lenient.png")

# %% [markdown]
# ## 5. Plot — edges

# %%
agg_edge = df_edge.groupby(['band', 'model'])[['tp', 'fn']].sum().reset_index()
agg_edge['n']        = agg_edge['tp'] + agg_edge['fn']
agg_edge['accuracy'] = agg_edge['tp'] / agg_edge['n']
agg_edge.to_csv(f'{nick_outlines}/df_edges_lenient_summary.csv', index=False)

counts_edge = (
    df_edge[df_edge['model'] == 'my_predictions_50']
    .groupby('band')[['tp', 'fn']].sum()
    .assign(total=lambda d: d['tp'] + d['fn'])
    .reindex(BANDS)
)
counts_edge['pct'] = counts_edge['total'] / counts_edge['total'].sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x = np.arange(len(BANDS))
for model, color in zip(MODELS, colors):
    df_m = agg_edge[agg_edge['model'] == model].set_index('band').reindex(BANDS)
    ax.plot(x, df_m['accuracy'].values, label=model_labels[model],
            color=color, marker='o', markersize=6)
ax.set_xticks(x)
ax.set_xticklabels(BANDS)
ax.invert_xaxis()
ax.set_ylabel('Accuracy (= recall among tree pixels)')
ax.set_title('Strict accuracy vs. distance inside treeline\n(QLD tiles only)')
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
bars = ax2.bar(BANDS, counts_edge['total'].values / 1e6, color='steelblue', width=0.7)
for bar, pct in zip(bars, counts_edge['pct'].values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)
ax2.invert_xaxis()
ax2.set_ylabel('Pixel count (millions)')
ax2.set_title('Tree pixels per band\n(% of classified tree pixels)')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Model accuracy inside the treeline — distance from patch boundary', y=1.02)
plt.tight_layout()
plt.savefig(f'{nick_outlines}/accuracy_vs_edge_lenient.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved accuracy_vs_edge_lenient.png")

# %% [markdown]
# ## 6. Plot — distances

# %%
agg_dist = df_dist.groupby(['distance', 'model'])[['tp', 'fp', 'fn', 'tn']].sum().reset_index()
agg_dist['n']        = agg_dist['tp'] + agg_dist['fp'] + agg_dist['fn'] + agg_dist['tn']
agg_dist['accuracy'] = (agg_dist['tp'] + agg_dist['tn']) / agg_dist['n']
agg_dist.to_csv(f'{nick_outlines}/df_distances_lenient_summary.csv', index=False)

plot_distances = list(range(1, 21))
counts_dist = (
    df_dist[(df_dist['model'] == 'my_predictions_50') & (df_dist['distance'] >= 1)]
    .groupby('distance')[['tp', 'fp', 'fn', 'tn']].sum()
    .assign(total=lambda d: d['tp'] + d['fp'] + d['fn'] + d['tn'])
    .reindex(plot_distances)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for model, color in zip(MODELS, colors):
    df_m = agg_dist[(agg_dist['model'] == model) & (agg_dist['distance'] >= 1)].sort_values('distance')
    fpr = np.maximum(1 - df_m['accuracy'].values, 1e-6)
    ax.plot(df_m['distance'], fpr, label=model_labels[model],
            color=color, marker='o', markersize=4)
ax.set_yscale('log')
ax.invert_yaxis()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda y, _: f'{1 - y:.4f}'.rstrip('0').rstrip('.')
))
ax.set_xlabel('Distance from treeline (pixels)')
ax.set_ylabel('Accuracy (log scale)')
ax.set_title('Strict accuracy vs. distance outside treeline\n(QLD tiles only)')
ax.set_xticks(plot_distances)
ax.set_xticklabels(plot_distances, fontsize=7)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
ax2.bar(plot_distances, counts_dist['total'].values / 1e6, color='steelblue', width=0.7)
ax2.set_yscale('log')
ax2.set_xlabel('Distance from treeline (pixels)')
ax2.set_ylabel('Pixel count (millions, log scale)')
ax2.set_title('Non-tree pixels per distance band\n(QLD tiles only)')
ax2.set_xticks(plot_distances)
ax2.set_xticklabels(plot_distances, fontsize=7)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Model accuracy vs. distance from the treeline (outside trees)', y=1.02)
plt.tight_layout()
plt.savefig(f'{nick_outlines}/accuracy_vs_distance_lenient.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved accuracy_vs_distance_lenient.png")
