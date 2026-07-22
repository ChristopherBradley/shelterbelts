"""Select the 'adjacent' evaluation tile set.

For each Nick-matched tile, pick ONE diagonally-adjacent lidar acquisition tile
(shares only a corner with the source tile, so minimal pixel overlap) from the
remaining lidar-only tiles. Prefer the diagonal neighbour that overlaps the
Nick training footprints the least, so the adjacent tile samples the same region
but as close to untrained ground as the acquisition grid allows.

Outputs a footprints gpkg + a plain list of .laz filepaths to feed lidar.py.
This only selects tiles; it does not run lidar.
"""
import os
import re
import argparse

import numpy as np
import geopandas as gpd

MATCHED = '/scratch/xe2/cb8590/Nick_matched_lidar_percentcover/nick_matched_lidar_footprints.gpkg'
LIDAR = '/g/data/xe2/cb8590/Outlines_Lidar/lidar_recent.gpkg'
NICK = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg'
OUT_GPKG = '/scratch/xe2/cb8590/Adjacent_eval_tiles/adjacent_eval_tiles.gpkg'
OUT_LIST = '/scratch/xe2/cb8590/Adjacent_eval_tiles/adjacent_laz_files.txt'

_PRODUCT_RE = re.compile(r'-([A-Za-z]+\d)-C\d+-')


def load_lidar():
    g = gpd.read_file(LIDAR)
    g['product'] = g['filepath'].apply(
        lambda p: (_PRODUCT_RE.search(os.path.basename(p)).group(1)
                   if _PRODUCT_RE.search(os.path.basename(p)) else None))
    g = g[~g['product'].fillna('').str.startswith('PHO')]           # lidar-only
    g['stub'] = g['filepath'].apply(lambda p: os.path.basename(p).rsplit('.', 1)[0])
    g = g[g['filepath'].apply(os.path.exists)]                       # actually downloaded
    return g.drop_duplicates('stub').to_crs(3577).reset_index(drop=True)


def main(max_overlap_source, dedupe):
    os.makedirs(os.path.dirname(OUT_GPKG), exist_ok=True)
    matched = gpd.read_file(MATCHED).to_crs(3577).reset_index(drop=True)
    lidar = load_lidar()
    nick_union = gpd.read_file(NICK).to_crs(3577).union_all()
    print(f'matched: {len(matched)}   lidar-only downloaded: {len(lidar)}', flush=True)

    sindex = lidar.sindex
    chosen, used = [], set()
    stats = {'no_candidate': 0}
    for _, m in matched.iterrows():
        mg = m.geometry
        mc = mg.centroid
        msize = np.sqrt(mg.area)
        # laz tiles whose bbox is near this matched tile
        near_idx = list(sindex.intersection(mg.buffer(1.5 * msize).bounds))
        best = None
        for j in near_idx:
            L = lidar.iloc[j]
            if dedupe and L['stub'] in used:
                continue
            inter = L.geometry.intersection(mg).area
            ov_src = inter / L.geometry.area
            if ov_src > max_overlap_source:        # skip tiles that sit on top of the source
                continue
            off = np.array([L.geometry.centroid.x - mc.x, L.geometry.centroid.y - mc.y])
            diagonal = (abs(off[0]) > 0.5 * msize) and (abs(off[1]) > 0.5 * msize)
            if not diagonal:
                continue
            ov_nick = L.geometry.intersection(nick_union).area / L.geometry.area
            # rank: least Nick overlap, then least source overlap
            key = (ov_nick, ov_src)
            if best is None or key < best[0]:
                best = (key, j, ov_src, ov_nick)
        if best is None:
            stats['no_candidate'] += 1
            continue
        _, j, ov_src, ov_nick = best
        L = lidar.iloc[j]
        used.add(L['stub'])
        chosen.append({'source_stub': m.get('filename', ''), 'laz_stub': L['stub'],
                       'filepath': L['filepath'], 'overlap_source': round(ov_src, 4),
                       'overlap_any_nick': round(ov_nick, 4), 'geometry': L.geometry})

    out = gpd.GeoDataFrame(chosen, crs=3577).to_crs(4326)
    out.to_file(OUT_GPKG, driver='GPKG')
    with open(OUT_LIST, 'w') as fh:
        fh.write('\n'.join(out['filepath']) + '\n')

    print(f'\nselected {len(out)} adjacent tiles   (no diagonal candidate: {stats["no_candidate"]})', flush=True)
    print(f'  overlap w/ source tile:  median {out.overlap_source.median():.3f}  max {out.overlap_source.max():.3f}', flush=True)
    print(f'  overlap w/ any Nick tile: median {out.overlap_any_nick.median():.3f}  '
          f'>50%: {(out.overlap_any_nick > 0.5).sum()}   ==0: {(out.overlap_any_nick == 0).sum()}', flush=True)
    print(f'saved {OUT_GPKG}\nsaved {OUT_LIST}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max_overlap_source', type=float, default=0.05,
                    help='Max fraction of the candidate tile that may overlap the source matched tile')
    ap.add_argument('--no_dedupe', action='store_true', help='Allow the same laz tile for multiple matched tiles')
    args = ap.parse_args()
    main(args.max_overlap_source, dedupe=not args.no_dedupe)
