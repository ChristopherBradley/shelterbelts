#!/usr/bin/env python
"""
evaluation_panel.py — unified 6-panel "Category & Distance Evaluations" figure.

One self-contained code path for every percent-cover lidar benchmark. Given a
named dataset config it accumulates category / edge / distance / precision-recall
statistics directly from the rasters (no intermediate summary scripts) and writes
the 6-panel PNG plus the raw CSVs.

Ground truth = lidar percent cover (10m, 0-100, 255=nodata). A pixel is "tree"
when cover >= cover_threshold (default 1%). Models compared, all reproject-matched
to the ground-truth grid:
  - my predictions (barra, one year) at confidence 50/60/70/80/90
  - Global Canopy Height v1 / v2 (height >= 1 m -> tree)
  - WorldCover (class 10 tree or 20 shrub -> tree)

The six panels:
  (0,0) recall vs distance inside the treeline (Edge 1..Core)
  (0,1) accuracy vs distance outside the treeline (log), to max_distance
  (0,2) non-tree pixel count per distance band
  (1,0) recall per tree category (strict)
  (1,1) precision vs recall across cover cutoffs 1-20%, one line per model
        (statics + my predictions at every confidence 50-90)
  (1,2) tree-pixel count per category (% of tree pixels)

Configs live in CONFIGS below. Run:  python evaluation_panel.py <dataset>
This supersedes evaluation_treecover.py, evaluation_shelter.py and the ad-hoc
accumulate_dist50 / plot_* panel scripts.
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------- config

PLOTS_DIR = '/scratch/xe2/cb8590/NSW_ag_treecover_10m/plots'
CSV_DIR = '/scratch/xe2/cb8590/NSW_ag_treecover_10m'
PRED_DIR_2020 = '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/subfolders'

CONFIGS = {
    # NEW NSW ag lidar benchmark (randomly sampled ag tiles, lidar percent-cover GT).
    'nsw_lidar': dict(
        out=f'{PLOTS_DIR}/08_presentation_nsw_lidar_c1.png',
        csv_prefix='nsw_lidar',
        title='Category & Distance Evaluations — NEW NSW ag lidar benchmark (cutoff=1, my pred 2020)',
        subtitle='NSW ag lidar',
        gt_dir='/scratch/xe2/cb8590/NSW_ag_treecover_10m', gt_suffix='_percentcover.tif',
        tiles_gpkg='/scratch/xe2/cb8590/NSW_ag_treecover_10m/eval_dataset_footprints.gpkg',
        stub_col='stub', subset_json=None,
        pred_dir=PRED_DIR_2020,
        gch1_dir='/scratch/xe2/cb8590/NSW_ag_GCH', gch2_dir='/scratch/xe2/cb8590/NSW_ag_GCH_v2',
        wc_dir='/scratch/xe2/cb8590/NSW_ag_worldcover',
        idx_e1='/scratch/xe2/cb8590/lidar_processing/eval_indices/e1',
        idx_e2='/scratch/xe2/cb8590/lidar_processing/eval_indices/e2',
        idx_e3='/scratch/xe2/cb8590/lidar_processing/eval_indices/e3',
        idx_windmethod='/scratch/xe2/cb8590/eval_indices_dist50/windmethod',
        max_distance=50, cover_threshold=1,
    ),
    # Lidar percent-cover GT cropped onto the exact Nick tile locations, built in
    # native UTM (no reproject onto the Nick grid) so it is pixel-registered like
    # 'adjacent' — the earlier Nick-grid version mis-registered GT ~1px and cost
    # the sharp height-thresholded GCH benchmark ~27pts of recall (verified).
    'matched': dict(
        out=f'{PLOTS_DIR}/10_presentation_matched_lidar_at_nick_locations.png',
        csv_prefix='matched',
        title='Category & Distance Evaluations — lidar GT cropped onto Nick tile locations, native UTM (my pred = barra 2020)',
        subtitle='lidar GT @ Nick locations',
        gt_dir='/scratch/xe2/cb8590/Matched_native_percentcover', gt_suffix='_percentcover.tif',
        tiles_gpkg='/scratch/xe2/cb8590/Matched_native_percentcover/matched_native_footprints.gpkg',
        stub_col='stub', subset_json=None,
        pred_dir=PRED_DIR_2020,
        gch1_dir='/scratch/xe2/cb8590/Matched_native_GCH', gch2_dir='/scratch/xe2/cb8590/Matched_native_GCH_v2',
        wc_dir='/scratch/xe2/cb8590/Matched_native_worldcover',
        idx_e1='/scratch/xe2/cb8590/matched_native_eval_indices/indices/e1',
        idx_e2='/scratch/xe2/cb8590/matched_native_eval_indices/indices/e2',
        idx_e3='/scratch/xe2/cb8590/matched_native_eval_indices/indices/e3',
        idx_windmethod='/scratch/xe2/cb8590/matched_native_eval_indices/indices/windmethod',
        max_distance=20, cover_threshold=1,
    ),
    # Lidar GT on tiles diagonally adjacent to the Nick-matched tiles: same region,
    # tiles the model did not train on. Isolates tile-level overfitting from region.
    'adjacent': dict(
        out=f'{PLOTS_DIR}/11_presentation_adjacent_to_nick.png',
        csv_prefix='adjacent',
        title='Category & Distance Evaluations — lidar GT on tiles diagonally adjacent to Nick locations (my pred = barra 2020)',
        subtitle='lidar GT @ adjacent tiles',
        gt_dir='/scratch/xe2/cb8590/Adjacent_eval_tiles', gt_suffix='_percentcover.tif',
        tiles_gpkg='/scratch/xe2/cb8590/Adjacent_eval_tiles/adjacent_eval_footprints.gpkg',
        stub_col='stub', subset_json=None,
        pred_dir=PRED_DIR_2020,
        gch1_dir='/scratch/xe2/cb8590/Adjacent_GCH', gch2_dir='/scratch/xe2/cb8590/Adjacent_GCH_v2',
        wc_dir='/scratch/xe2/cb8590/Adjacent_worldcover',
        idx_e1='/scratch/xe2/cb8590/adjacent_eval_indices/indices/e1',
        idx_e2='/scratch/xe2/cb8590/adjacent_eval_indices/indices/e2',
        idx_e3='/scratch/xe2/cb8590/adjacent_eval_indices/indices/e3',
        idx_windmethod='/scratch/xe2/cb8590/adjacent_eval_indices/indices/windmethod',
        max_distance=20, cover_threshold=1,
        # Blank out any GT pixels that fall inside a Nick training tile (the ~23
        # adjacent tiles that clip a Nick footprint) so training pixels can't leak in.
        nick_mask_gpkg='/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg',
    ),
}

# ----------------------------------------------------------------------------- constants

CATEGORIES = [11, 12, 13, 15, 17, 18, 19]
CATEGORY_LABELS = {11: 'Scattered Trees', 12: 'Patch Core', 13: 'Patch Edge', 15: 'Trees in Gullies',
                   17: 'Trees next to Roads', 18: 'Linear Patches', 19: 'Non-linear Patches'}
BANDS = ['Edge 1', 'Edge 2', 'Edge 3', 'Core (4+)']
CONFS = [50, 60, 70, 80, 90]
CUTOFFS_PR = [1, 5, 10, 15, 20]

MY_MODELS = [f'my_predictions_{c}' for c in CONFS]
STATICS = ['global_canopy_height_v1_trees', 'global_canopy_height_v2_trees', 'worldcover_trees']
MODELS = STATICS + MY_MODELS
MODEL_LABELS = {'global_canopy_height_v1_trees': 'GCH v1', 'global_canopy_height_v2_trees': 'GCH v2',
                'worldcover_trees': 'WorldCover', 'my_predictions_50': 'My pred (0.50)',
                'my_predictions_60': 'My pred (0.60)', 'my_predictions_70': 'My pred (0.70)',
                'my_predictions_80': 'My pred (0.80)', 'my_predictions_90': 'My pred (0.90)'}
COLORS = {'global_canopy_height_v1_trees': '#ff7f0e', 'global_canopy_height_v2_trees': '#2ca02c',
          'worldcover_trees': '#1f77b4', 'my_predictions_50': '#fcbba1', 'my_predictions_60': '#fc9272',
          'my_predictions_70': '#fb6a4a', 'my_predictions_80': '#de2d26', 'my_predictions_90': '#99000d'}

# ----------------------------------------------------------------------------- raster helpers


def load_da(path, ref):
    da = rxr.open_rasterio(path).isel(band=0).drop_vars('band')
    return da.rio.reproject_match(ref, resampling=Resampling.nearest).astype(int)


def windowed_da(path, bbox_3857):
    with rasterio.open(path) as s:
        w = from_bounds(*bbox_3857, s.transform)
        data = s.read(1, window=w)
        t = s.window_transform(w)
    h, wd = data.shape
    if h == 0 or wd == 0:
        return None
    x = t.c + (np.arange(wd) + 0.5) * t.a
    y = t.f + (np.arange(h) + 0.5) * t.e
    return xr.DataArray(data, dims=('y', 'x'), coords={'y': y, 'x': x}).rio.write_crs('EPSG:3857')


def build_pred_index(pred_dir):
    recs = []
    for f in os.listdir(pred_dir):
        if f.endswith('_merged_predicted.tif'):
            with rasterio.open(os.path.join(pred_dir, f)) as s:
                b = s.bounds
            recs.append({'path': os.path.join(pred_dir, f), 'geometry': box(b.left, b.bottom, b.right, b.top)})
    return gpd.GeoDataFrame(recs, crs='EPSG:3857')


def tile_stubs_and_bounds(cfg):
    """Return (stubs, bounds_3857) where bounds maps stub -> (minx,miny,maxx,maxy)."""
    gdf = gpd.read_file(cfg['tiles_gpkg'])
    if cfg['stub_col']:
        gdf['stub'] = gdf[cfg['stub_col']]
    else:
        gdf['stub'] = [f.split('.')[0] for f in gdf['filename']]
    if cfg['subset_json']:
        keep = set(json.load(open(cfg['subset_json'])))
        gdf = gdf[gdf['stub'].isin(keep)]
    gdf = gdf.drop_duplicates('stub').to_crs('EPSG:3857')
    bounds = {r['stub']: r.geometry.bounds for _, r in gdf.iterrows()}
    return list(gdf['stub']), bounds

# ----------------------------------------------------------------------------- accumulation


def process(cfg):
    max_d = cfg['max_distance']
    distances = list(range(0, max_d + 1))
    ct = cfg['cover_threshold']

    stubs, bounds = tile_stubs_and_bounds(cfg)
    nick_mask = gpd.read_file(cfg['nick_mask_gpkg']).to_crs(4326) if cfg.get('nick_mask_gpkg') else None
    if nick_mask is not None:
        nick_sindex = nick_mask.sindex
    pidx = build_pred_index(cfg['pred_dir'])
    stub_geom = gpd.GeoDataFrame({'stub': list(bounds)},
                                 geometry=[box(*bounds[s]) for s in bounds], crs='EPSG:3857')
    pj = gpd.sjoin(stub_geom, pidx, how='left', predicate='intersects') \
        .drop_duplicates('stub').set_index('stub')['path']

    cat_rows, edge_rows, dist_rows = [], [], []
    overall = {(m, c): [0, 0, 0, 0] for m in MODELS for c in CUTOFFS_PR}
    n = len(stubs)
    n_used = 0
    for i, stub in enumerate(stubs):
        paths = {
            'gt': f"{cfg['gt_dir']}/{stub}{cfg['gt_suffix']}",
            'wc': f"{cfg['wc_dir']}/{stub}_worldcover.tif",
            'g1': f"{cfg['gch1_dir']}/{stub}_canopy_height.tif",
            'g2': f"{cfg['gch2_dir']}/{stub}_canopy_height.tif",
            'e1': f"{cfg['idx_e1']}/{stub}_linear_categories.tif",
            'e2': f"{cfg['idx_e2']}/{stub}_linear_categories.tif",
            'tc': f"{cfg['idx_e3']}/{stub}_tree_categories.tif",
            'sd': f"{cfg['idx_windmethod']}/{stub}_shelter_distances.tif"}
        if not all(os.path.exists(p) for p in paths.values()):
            continue

        gt = rxr.open_rasterio(paths['gt']).isel(band=0).drop_vars('band')
        pc = gt.values
        valid = pc != 255
        # Exclude GT pixels inside any Nick training tile (prevents training-pixel leakage).
        if nick_mask is not None:
            tb4326 = gpd.GeoSeries([box(*gt.rio.bounds())], crs=gt.rio.crs).to_crs(4326).iloc[0]
            cand = nick_mask.iloc[list(nick_sindex.query(tb4326, predicate='intersects'))]
            if len(cand):
                sub = cand.to_crs(gt.rio.crs)
                inside = geometry_mask(sub.geometry, out_shape=pc.shape,
                                       transform=gt.rio.transform(), invert=True)
                valid = valid & ~inside
        if valid.sum() == 0:
            continue
        n_used += 1
        wc = rxr.open_rasterio(paths['wc']).isel(band=0).drop_vars('band') \
            .rio.reproject_match(gt, resampling=Resampling.nearest).values
        g1 = load_da(paths['g1'], gt).values
        g2 = load_da(paths['g2'], gt).values
        e1 = load_da(paths['e1'], gt).values.ravel()
        e2 = load_da(paths['e2'], gt).values.ravel()
        tc = load_da(paths['tc'], gt).values.ravel()
        sd = load_da(paths['sd'], gt).values.ravel()

        preds = {'global_canopy_height_v1_trees': (g1 >= 1),
                 'global_canopy_height_v2_trees': (g2 >= 1),
                 'worldcover_trees': ((wc == 10) | (wc == 20))}
        path = pj.get(stub)
        if isinstance(path, str):
            da = windowed_da(path, bounds[stub])
            if da is not None:
                pv = da.rio.reproject_match(gt, resampling=Resampling.nearest).values
                for c in CONFS:
                    preds[f'my_predictions_{c}'] = pv >= c

        tree_mask = (pc >= ct) & valid
        tm = tree_mask.ravel()
        v = valid

        # overall precision-recall at multiple cover cutoffs
        for cutoff in CUTOFFS_PR:
            yt = ((pc >= cutoff) & v)[v]
            for model in MODELS:
                if model not in preds:
                    continue
                yp = preds[model][v]
                a = overall[(model, cutoff)]
                a[0] += int((yt & yp).sum()); a[1] += int((~yt & yp).sum())
                a[2] += int((yt & ~yp).sum()); a[3] += int((~yt & ~yp).sum())

        # categories (strict), labelled by edge1 category
        for cat in CATEGORIES:
            m = tm & (e1 == cat)
            if m.sum() == 0:
                continue
            for model in MODELS:
                if model not in preds:
                    continue
                ys = preds[model].ravel()[m]
                cat_rows.append({'category': cat, 'model': model, 'tp': int(ys.sum()), 'fn': int((~ys).sum())})

        # edges (strict)
        band_masks = {'Edge 1': tm & (e1 == 13), 'Edge 2': tm & (e1 != 13) & (e2 == 13),
                      'Edge 3': tm & (e2 != 13) & (tc == 13), 'Core (4+)': tm & (tc == 12)}
        for band, m in band_masks.items():
            if m.sum() == 0:
                continue
            for model in MODELS:
                if model not in preds:
                    continue
                yp = preds[model].ravel()[m]
                edge_rows.append({'band': band, 'model': model, 'tp': int(yp.sum()), 'fn': int((~yp).sum())})

        # distances outside the treeline (strict)
        for d in distances:
            m = (sd == d) if d == 0 else ((sd == d) & (~tm))
            if m.sum() == 0:
                continue
            yt = tm[m]
            for model in MODELS:
                if model not in preds:
                    continue
                yp = preds[model].ravel()[m]
                dist_rows.append({'distance': d, 'model': model,
                                  'tp': int((yt & yp).sum()), 'fp': int(((~yt) & yp).sum()),
                                  'fn': int((yt & (~yp)).sum()), 'tn': int(((~yt) & (~yp)).sum())})
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{n}', flush=True)

    overall_rows = []
    for (m, c), (tp, fp, fn, tn) in overall.items():
        tot = tp + fp + fn + tn
        overall_rows.append({'model': m, 'cutoff': c,
                             'precision': tp / (tp + fp) if tp + fp else np.nan,
                             'recall': tp / (tp + fn) if tp + fn else np.nan,
                             'accuracy': (tp + tn) / tot if tot else np.nan})
    return (pd.DataFrame(cat_rows), pd.DataFrame(edge_rows), pd.DataFrame(dist_rows),
            pd.DataFrame(overall_rows), n_used)

# ----------------------------------------------------------------------------- plotting


def recall_agg(df, key):
    a = df.groupby([key, 'model'])[['tp', 'fn']].sum().reset_index()
    a['accuracy'] = a['tp'] / (a['tp'] + a['fn'])
    return a


def plot(cfg, cat, edge, dist, overall, n_tiles):
    sub = cfg['subtitle']
    max_d = cfg['max_distance']
    pd_ = list(range(1, max_d + 1))
    xticks = pd_ if max_d <= 20 else list(range(0, max_d + 1, 5))

    agg_cat = recall_agg(cat, 'category')
    agg_edge = recall_agg(edge, 'band')
    agg_dist = dist.groupby(['distance', 'model'])[['tp', 'fp', 'fn', 'tn']].sum().reset_index()
    agg_dist['accuracy'] = (agg_dist.tp + agg_dist.tn) / agg_dist[['tp', 'fp', 'fn', 'tn']].sum(axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    # (0,0) edges inside treeline
    ax = axes[0, 0]; x = np.arange(len(BANDS))
    for model in MODELS:
        dm = agg_edge[agg_edge.model == model].set_index('band').reindex(BANDS)
        ax.plot(x, dm['accuracy'].values, marker='o', markersize=6, label=MODEL_LABELS[model], color=COLORS[model])
    ax.set_xticks(x); ax.set_xticklabels(BANDS); ax.invert_xaxis(); ax.set_ylim(0, 1)
    ax.set_ylabel('Recall among tree pixels'); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f'Model accuracy vs. distance inside treeline\n({sub}, cutoff={cfg["cover_threshold"]})')

    # (0,1) distance outside treeline (log)
    ax = axes[0, 1]
    for model in MODELS:
        dm = agg_dist[(agg_dist.model == model) & (agg_dist.distance >= 1)].sort_values('distance')
        ax.plot(dm['distance'], np.maximum(1 - dm['accuracy'].values, 1e-6), marker='o', markersize=4,
                label=MODEL_LABELS[model], color=COLORS[model])
    ax.set_yscale('log'); ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{1-y:.4f}'.rstrip('0').rstrip('.')))
    ax.set_xlabel('Distance from treeline (pixels)'); ax.set_ylabel('Accuracy (log scale)')
    ax.set_xticks(xticks); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f'Model accuracy vs. distance outside treeline\n({sub}, to {max_d}px)')

    # (0,2) non-tree pixels per distance band
    ax = axes[0, 2]
    counts = (dist[(dist.model == 'my_predictions_50') & (dist.distance >= 1)]
              .groupby('distance')[['tp', 'fp', 'fn', 'tn']].sum()
              .assign(total=lambda d: d.tp + d.fp + d.fn + d.tn).reindex(pd_))
    ax.bar(pd_, counts['total'].values / 1e6, color='steelblue', width=0.8 if max_d <= 20 else 0.7)
    ax.set_yscale('log'); ax.set_xlabel('Distance from treeline (pixels)')
    ax.set_ylabel('Pixel count (millions, log scale)'); ax.set_xticks(xticks)
    ax.grid(axis='y', alpha=0.3); ax.set_title(f'Non-tree pixels per distance band\n({sub})')

    # (1,0) category strict
    ax = axes[1, 0]; x = np.arange(len(CATEGORIES)); w = 0.8 / len(MODELS)
    for j, model in enumerate(MODELS):
        dm = agg_cat[agg_cat.model == model].set_index('category').reindex(CATEGORIES)
        ax.bar(x + (j - len(MODELS) / 2 + 0.5) * w, dm['accuracy'].values, width=w,
               label=MODEL_LABELS[model], color=COLORS[model])
    ax.set_xticks(x); ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Recall among tree pixels'); ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f'Model accuracy per tree category\n(strict — {sub}, cutoff={cfg["cover_threshold"]})')

    # (1,1) precision-recall — every model a line across cover cutoffs 1-20%
    ax = axes[1, 1]
    for model in STATICS + MY_MODELS:
        s = overall[overall.model == model].set_index('cutoff').reindex(CUTOFFS_PR).reset_index()
        ax.plot(s.recall * 100, s.precision * 100, color=COLORS[model], lw=2.2, marker='o', ms=6,
                label=MODEL_LABELS[model], zorder=3)
        for _, r in s.iterrows():
            ax.annotate(f'{int(r.cutoff)}', (r.recall * 100, r.precision * 100),
                        textcoords='offset points', xytext=(5, 4), fontsize=6.5, color='#888')
    ax.set_xlabel('Recall (%)'); ax.set_ylabel('Precision (%)'); ax.margins(0.13); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.set_title(f'Precision vs. recall across cover cutoffs 1-20%\n({sub}; labelled at each cutoff)')

    # (1,2) tree pixels per category (%)
    ax = axes[1, 2]
    cc = (cat[cat.model == 'my_predictions_50'].groupby('category')[['tp', 'fn']].sum()
          .assign(total=lambda d: d.tp + d.fn).reindex(CATEGORIES).fillna(0))
    cc['pct'] = cc['total'] / cc['total'].sum() * 100
    x = np.arange(len(CATEGORIES))
    bars = ax.bar(x, cc['total'].values / 1e6, color='steelblue', width=0.7)
    for bar, pct in zip(bars, cc['pct'].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1, f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Pixel count (millions, log scale)'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Tree pixels per category\n(% of total tree pixels)')

    fig.suptitle(f'{cfg["title"]}  (n={n_tiles})', fontsize=15, y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)
    plt.savefig(cfg['out'], dpi=150, bbox_inches='tight')
    print(f'Saved {cfg["out"]}', flush=True)


def run(dataset):
    cfg = CONFIGS[dataset]
    cat, edge, dist, overall, n_tiles = process(cfg)

    p = cfg['csv_prefix']
    cat.to_csv(f'{CSV_DIR}/{p}_categories_raw.csv', index=False)
    edge.to_csv(f'{CSV_DIR}/{p}_edges_raw.csv', index=False)
    dist.to_csv(f'{CSV_DIR}/{p}_distances_raw.csv', index=False)
    overall.to_csv(f'{CSV_DIR}/{p}_overall_pr.csv', index=False)

    plot(cfg, cat, edge, dist, overall, n_tiles)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', choices=list(CONFIGS), help='Which benchmark to evaluate')
    args = ap.parse_args()
    run(args.dataset)
