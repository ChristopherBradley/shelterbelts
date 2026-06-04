import os
import glob
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping

from shelterbelts.indices.shelter_metrics import linear_categories_labels

# --- Paths (change these to switch datasets) ---
tif_folder = '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/expanded_grazing_no_bwh'
tif_suffix  = 'default_percentmethod_merged_y2020_predicted_expanded20_linear_categories.tif'
states_shp  = '/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp'
output_dir  = '/scratch/xe2/cb8590/linear_categories_stats'
state_col   = 'STE_NAME21'
nodata      = 255


def _process_tif(tif_path, state_records):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        height, width = data.shape
        bounds = src.bounds

    valid = data != nodata
    global_counts = np.bincount(data[valid].astype(np.intp), minlength=256)

    tile_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    state_id_map = {}
    shapes = []
    for i, (state_name, geom) in enumerate(state_records, start=1):
        if not tile_box.intersects(geom):
            continue
        try:
            clipped = geom.intersection(tile_box)
        except Exception:
            clipped = geom
        if clipped.is_empty:
            continue
        state_id_map[i] = state_name
        shapes.append((mapping(clipped), i))

    state_counts = {}
    if shapes:
        state_arr = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        for sid, state_name in state_id_map.items():
            pixels = data[(state_arr == sid) & valid]
            if len(pixels) > 0:
                state_counts[state_name] = np.bincount(pixels.astype(np.intp), minlength=256)
        del state_arr

    return global_counts, state_counts


def aggregate(results):
    global_total = np.zeros(256, dtype=np.int64)
    state_totals = {}
    for global_counts, state_counts in results:
        global_total += global_counts
        for state, counts in state_counts.items():
            state_totals.setdefault(state, np.zeros(256, dtype=np.int64))
            state_totals[state] += counts

    used = [c for c in range(256) if global_total[c] > 0]
    global_df = pd.DataFrame({
        'category_id':   used,
        'category_name': [linear_categories_labels.get(c, str(c)) for c in used],
        'count':         global_total[used],
    })
    global_df['percent'] = 100 * global_df['count'] / global_df['count'].sum()
    global_df = global_df.sort_values('count', ascending=False).reset_index(drop=True)

    rows = []
    for state, counts in sorted(state_totals.items()):
        total = int(counts.sum())
        for cat in range(256):
            if counts[cat] > 0:
                rows.append({
                    'state':            state,
                    'category_id':      cat,
                    'category_name':    linear_categories_labels.get(cat, str(cat)),
                    'count':            int(counts[cat]),
                    'percent_of_state': 100 * counts[cat] / total,
                })
    state_df = pd.DataFrame(rows)
    return global_df, state_df


def run(tif_folder, tif_suffix, states_shp, output_dir, state_col, limit=None):
    os.makedirs(output_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(tif_folder, f'*{tif_suffix}')))
    print(f'Found {len(tif_files)} TIF files')
    if limit is not None:
        tif_files = tif_files[:limit]
        print(f'Limiting to {len(tif_files)} files (--limit {limit})')

    with rasterio.open(tif_files[0]) as src:
        tif_epsg = src.crs.to_epsg()

    states_gdf = gpd.read_file(states_shp).to_crs(epsg=tif_epsg)

    if state_col not in states_gdf.columns:
        states_gdf[state_col] = [f'Region_{i}' for i in range(len(states_gdf))]
    else:
        states_gdf = states_gdf.query(
            f"{state_col} not in ['Outside Australia', 'Other Territories']"
        )

    state_records = [(row[state_col], row.geometry) for _, row in states_gdf.iterrows()]

    results = [_process_tif(f, state_records) for f in tif_files]

    global_df, state_df = aggregate(results)

    global_csv = os.path.join(output_dir, 'linear_categories_global.csv')
    state_csv  = os.path.join(output_dir, 'linear_categories_by_state.csv')
    global_df.to_csv(global_csv, index=False)
    state_df.to_csv(state_csv, index=False)
    print(f'Saved: {global_csv}')
    print(f'Saved: {state_csv}')
    print(global_df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count linear_categories pixels globally and per state.')
    parser.add_argument('--tif_folder',  default=tif_folder)
    parser.add_argument('--tif_suffix',  default=tif_suffix)
    parser.add_argument('--states_shp',  default=states_shp)
    parser.add_argument('--output_dir',  default=output_dir)
    parser.add_argument('--state_col',   default=state_col)
    parser.add_argument('--limit',       type=int, default=None,
                        help='Process only the first N TIF files (for testing)')
    args = parser.parse_args()
    run(args.tif_folder, args.tif_suffix, args.states_shp,
        args.output_dir, args.state_col, args.limit)
