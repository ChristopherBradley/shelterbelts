"""GCH v2 -> 10 m percent-cover + canopy-height tiles, aligned to a global 10 m lattice.

Two phases for the full-ag indices run:

  bin_gch_tile   : stream one 1 m GCH v2 30 km tile into 10 m percent-cover + mean-tree-height rasters,
                   snapped to a GLOBAL 10 m EPSG:3857 lattice (origin at multiples of `res` from 0) so
                   every binned tile — and every 4 km tile cut from them — shares one seamless grid.
  cut_region     : for one ~200 km region, cut each barra_bboxs_ag 4 km tile (expanded by `overlap_px`
                   for the shelter border) out of the binned mosaic and save percent/ + height/ tiles
                   named by the barra coord stub (e.g. 34_45-148_53), matching the existing pipeline.

Percent cell = 100 * (#GCH pixels >= height_threshold m) / (#GCH pixels in cell); a downstream
cover_threshold of 1 then treats any cell with >=1% cover as tree. Height cell = mean height of the
tree pixels in the cell (metres, clipped 0-60), used by the height-aware shelter stage.
"""
import os
import glob
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
import rioxarray as rxr
from rioxarray.merge import merge_arrays


def _snapped_grid(bounds, res):
    """Global-lattice 10 m grid covering `bounds` (left, bottom, right, top)."""
    left, bottom, right, top = bounds
    x0 = math.floor(left / res) * res
    y0 = math.ceil(top / res) * res            # top edge
    nx = int(math.ceil((right - x0) / res))
    ny = int(math.ceil((y0 - bottom) / res))
    return x0, y0, nx, ny


def bin_gch_tile(gch_tif, out_percent, out_height, res=10.0, height_threshold=1.0, block_rows=2048):
    """Stream a 1 m GCH tile into global-lattice 10 m percent + mean-tree-height uint8 rasters."""
    with rasterio.open(gch_tif) as src:
        crs = src.crs
        src_h, src_w = src.height, src.width
        a, e = src.transform.a, src.transform.e
        left, top = src.transform.c, src.transform.f

        x0, y0, nx, ny = _snapped_grid(src.bounds, res)
        out_transform = from_origin(x0, y0, res, res)

        col_centres = left + (np.arange(src_w) + 0.5) * a
        tcol = np.clip(((col_centres - x0) / res).astype(np.int64), 0, nx - 1)

        tree_count = np.zeros(ny * nx, dtype=np.float64)
        total_count = np.zeros(ny * nx, dtype=np.float64)
        height_sum = np.zeros(ny * nx, dtype=np.float64)

        for r0 in range(0, src_h, block_rows):
            nrows = min(block_rows, src_h - r0)
            arr = src.read(1, window=Window(0, r0, src_w, nrows)).astype(np.float32)
            row_centres = top + (np.arange(r0, r0 + nrows) + 0.5) * e
            trow = np.clip(((y0 - row_centres) / res).astype(np.int64), 0, ny - 1)
            idx = (trow[:, None] * nx + tcol[None, :]).ravel()
            tree = (arr >= height_threshold)
            total_count += np.bincount(idx, minlength=ny * nx)
            tree_count += np.bincount(idx, weights=tree.ravel().astype(np.float64), minlength=ny * nx)
            height_sum += np.bincount(idx, weights=(arr * tree).ravel().astype(np.float64), minlength=ny * nx)

    with np.errstate(invalid='ignore', divide='ignore'):
        percent = np.where(total_count > 0, np.round(tree_count / total_count * 100), 0)
        height = np.where(tree_count > 0, np.round(height_sum / tree_count), 0)
    percent = np.clip(percent, 0, 100).astype(np.uint8).reshape(ny, nx)
    height = np.clip(height, 0, 60).astype(np.uint8).reshape(ny, nx)

    for path, arr in ((out_percent, percent), (out_height, height)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with rasterio.open(path, 'w', driver='GTiff', height=ny, width=nx, count=1,
                           dtype='uint8', crs=crs, transform=out_transform,
                           compress='deflate', tiled=True, blockxsize=512, blockysize=512) as dst:
            dst.write(arr, 1)
    return out_percent, out_height


def bin_gch_tiles(gch_tifs, binned_dir, res=10.0, height_threshold=1.0, skip_existing=True):
    """Bin a list of GCH tiles into binned_dir/percent + binned_dir/height (skips ones already done)."""
    pdir, hdir = os.path.join(binned_dir, 'percent'), os.path.join(binned_dir, 'height')
    n = 0
    for t in gch_tifs:
        stem = os.path.splitext(os.path.basename(t))[0]
        op, oh = os.path.join(pdir, f'{stem}.tif'), os.path.join(hdir, f'{stem}.tif')
        if skip_existing and os.path.exists(op) and os.path.exists(oh):
            continue
        bin_gch_tile(t, op, oh, res=res, height_threshold=height_threshold)
        n += 1
    return n


def combine_centrelines(folder, out_gpkg, suffix='_centrelines.gpkg'):
    """Concatenate all per-tile ``*_centrelines.gpkg`` in a folder into one region GeoPackage,
    stamping each row with its source ``tile`` (from the filename), mirroring combine_patch_metrics_csvs."""
    files = sorted(f for f in glob.glob(os.path.join(folder, f'*{suffix}'))
                   if os.path.abspath(f) != os.path.abspath(out_gpkg))
    if not files:
        print(f"No '*{suffix}' files in {folder}")
        return None
    parts = []
    for f in files:
        g = gpd.read_file(f)
        g.insert(0, 'tile', os.path.basename(f)[:-len(suffix)])
        parts.append(g)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    out.to_file(out_gpkg, driver='GPKG')
    print(f"Combined {len(files)} gpkgs ({len(out)} lines) -> {out_gpkg}")
    return out_gpkg


def region_tiles(bboxs_gdf, region):
    """Return (sub_gdf_in_3857, stubs) for the ag tiles whose centroid falls in `region` (lat_X_lon_Y)."""
    import math as _m
    c = bboxs_gdf.geometry.centroid
    stubs = np.array([f'{y:.2f}-{x:.2f}'.replace('.', '_')[1:] for y, x in zip(c.y.values, c.x.values)])
    regs = np.array([f'lat_{int(_m.floor(abs(y) / 2) * 2)}_lon_{int(_m.floor(x / 2) * 2)}'
                     for y, x in zip(c.y.values, c.x.values)])
    mask = regs == region
    sub = bboxs_gdf[mask].to_crs('EPSG:3857').reset_index(drop=True)
    return sub, stubs[mask]


def _place_from_tiles(win_bounds, tile_paths, res):
    """Stitch a window from same-lattice binned tiles via rasterio windowed reads (low memory).

    All binned tiles share one global res-metre lattice, so a window maps to each tile by exact
    integer pixel offsets — no resampling, no full-tile loads, no growing cache. Returns the uint8
    window array (zeros where no tile covers), or None if nothing overlaps.
    """
    wx0, wy_bot, wx1, wy_top = win_bounds
    W = int(round((wx1 - wx0) / res))
    H = int(round((wy_top - wy_bot) / res))
    if W <= 0 or H <= 0:
        return None
    out = np.zeros((H, W), dtype=np.uint8)
    covered = False
    for tp in tile_paths:
        with rasterio.open(tp) as src:
            tlx, tby, trx, tty = src.bounds
            ox0, ox1 = max(wx0, tlx), min(wx1, trx)
            oy0, oy1 = max(wy_bot, tby), min(wy_top, tty)
            if ox0 >= ox1 or oy0 >= oy1:
                continue
            col0 = int(round((ox0 - tlx) / res)); col1 = int(round((ox1 - tlx) / res))
            row0 = int(round((tty - oy1) / res)); row1 = int(round((tty - oy0) / res))
            data = src.read(1, window=Window(col0, row0, col1 - col0, row1 - row0))
            oc0 = int(round((ox0 - wx0) / res)); orr0 = int(round((wy_top - oy1) / res))
            out[orr0:orr0 + data.shape[0], oc0:oc0 + data.shape[1]] = data
            covered = True
    return out if covered else None


def cut_region(region, bboxs_gdf, binned_dir, footprints_gdf, out_dir, overlap_px=50, res=10.0,
               save_height=True):
    """Cut every ag 4 km tile in `region` (expanded by overlap_px) from the binned 10 m tiles.

    Uses windowed rasterio reads (see _place_from_tiles) so memory stays flat regardless of how many
    binned GCH tiles the region spans — the previous rioxarray cache + merge_arrays approach OOM'd at
    4 GB on the large 2500-tile regions.

    save_height=False skips the height/ output for runs that don't use height-aware shelter (the
    binned height tiles are still required to exist, so tile selection is identical either way).
    """
    from shapely.geometry import box as _box
    sub, stubs = region_tiles(bboxs_gdf, region)
    pdir, hdir = os.path.join(out_dir, 'percent'), os.path.join(out_dir, 'height')
    os.makedirs(pdir, exist_ok=True)
    if save_height:
        os.makedirs(hdir, exist_ok=True)
    binned_p = os.path.join(binned_dir, 'percent')
    binned_h = os.path.join(binned_dir, 'height')
    fp_sindex = footprints_gdf.sindex
    fp = footprints_gdf.reset_index(drop=True)
    ov = overlap_px * res

    n = 0
    for geom, stub in zip(sub.geometry, stubs):
        minx, miny, maxx, maxy = geom.bounds
        # Snap the expanded window to the global lattice so it lines up pixel-exactly with the tiles.
        wx0 = math.floor((minx - ov) / res) * res
        wx1 = math.ceil((maxx + ov) / res) * res
        wy_bot = math.floor((miny - ov) / res) * res
        wy_top = math.ceil((maxy + ov) / res) * res
        win = _box(wx0, wy_bot, wx1, wy_top)
        hit_idx = fp_sindex.query(win, predicate='intersects')
        p_paths, h_paths = [], []
        for i in hit_idx:
            stem = os.path.splitext(fp.iloc[i]['filename'])[0]
            bp = os.path.join(binned_p, f'{stem}.tif')
            bh = os.path.join(binned_h, f'{stem}.tif')
            if os.path.exists(bp) and os.path.exists(bh):
                p_paths.append(bp); h_paths.append(bh)
        if not p_paths:
            continue
        out_p = _place_from_tiles((wx0, wy_bot, wx1, wy_top), p_paths, res)
        if out_p is None:
            continue
        outputs = [(os.path.join(pdir, f'{stub}.tif'), out_p)]
        if save_height:
            outputs.append((os.path.join(hdir, f'{stub}.tif'),
                            _place_from_tiles((wx0, wy_bot, wx1, wy_top), h_paths, res)))
        transform = from_origin(wx0, wy_top, res, res)
        for path, arr in outputs:
            with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1],
                               count=1, dtype='uint8', crs='EPSG:3857', transform=transform,
                               compress='deflate') as dst:
                dst.write(arr, 1)
        n += 1
    return n, len(sub)
