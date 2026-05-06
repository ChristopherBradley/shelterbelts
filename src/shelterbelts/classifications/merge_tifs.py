# Merge a folder of GeoTIFF tiles into a single uint8 raster
import glob
import os
import argparse
import re

import rioxarray as rxr
import pandas as pd
import numpy as np


from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.utils.tiles import merge_tiles_bbox, merged_ds


def extract_year(name):
    """Extract a 6-character substring starting at the first digit."""
    first_number = re.search(r'\d', name).start()
    year = name[first_number:first_number + 6]    # YYYYMM or YYYY__ I'm just using this to sort the year-month, so it doesn't matter if it's formatted perfectly (lots will have an extra 2 digits that can be ignored)
    return year


def cluster_bounds(bounds, tolerance=0.02):
    """Find tiles with matching bounds (or at least within the tolerance)"""
    groups = -np.ones(len(bounds), dtype=int)
    group_id = 0
    for i in range(len(bounds)):
        if groups[i] != -1:
            continue
        diffs = np.abs(bounds - bounds.iloc[i])
        mask = (diffs <= tolerance).all(axis=1)
        groups[mask] = group_id
        group_id += 1
    return groups


def merge_tifs(base_dir, tmpdir='/tmp', suffix='.tif', subdir='', crs=None, dont_reproject=False, dedup=False):
    """
    Merge a folder of GeoTIFF tiles into a single uint8 raster.

    Parameters
    ----------
    base_dir : str
        Directory containing the folder of tif files to be merged.
    tmpdir : str, optional
        Directory for temporary files.
    suffix : str, optional
        Glob suffix matching the files to merge.
    subdir : str, optional
        Subfolder inside base_dir that contains the tifs. Pass an empty
        string if the tifs sit directly in base_dir.
    crs : str, optional
        Force the output to a specific CRS (e.g. "EPSG:3857"). If None, a
        UTM CRS is estimated from a sample tile.
    dont_reproject : bool, optional
        Skip per-tile reprojection during the uint8 conversion step. Useful
        when all tiles are already in the same CRS.
    dedup : bool, optional
        When True, tiles with matching footprints but different dates are
        deduplicated to keep only the most recent.

    Returns
    -------
    xarray.DataArray
        The merged raster, reprojected to crs (or the estimated UTM CRS).

    Notes
    -----
    Writes to disk:
        - a subfolder with each tif copied as uint8 (if they weren't already uint8)
        - footprints.gpkg  (using bounding_boxes.py)
        - merged.tif

    """
    glob_path = os.path.join(base_dir, subdir, f'*{suffix}')
    filenames = glob.glob(glob_path)

    # Use the middle filename to choose a crs.
    da = rxr.open_rasterio(filenames[len(filenames)//2]).isel(band=0).drop_vars("band")
    if not crs:
        final_crs = da.rio.estimate_utm_crs()
    else:
        final_crs = crs
    print(f"Merging with crs: {final_crs}")

    suffix_stub = suffix.split('.')[0]

    if da.dtype == 'uint8':
        outdir = base_dir   # Don't convert to uint8
    else:
        outdir = os.path.join(base_dir, f'uint8{suffix_stub}')

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        for i, filename in enumerate(filenames):
            da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
            da = da.where(da < 100, 100)    # Truncate trees > 100m since these are more likely to be data errors in Australia.
            da = da.where(da != -9999, 255) # Remap nodata into the uint8 range.
            da = da.rio.write_nodata(255)
            da = da.astype('uint8')

            if not dont_reproject:
                da = da.rio.reproject(final_crs)

            outfile = f"{filename.split('/')[-1].split('.')[0]}_uint8.tif"
            outpath = os.path.join(outdir, outfile)
            da.rio.to_raster(outpath, compress="lzw")
            if i % 100 == 0:
                print(f"Saved {i}/{len(filenames)}:", outpath)

    stub = f"{'_'.join(outdir.split('/')[-2:]).split('.')[0]}_{suffix_stub}"
    gdf = bounding_boxes(outdir, crs=final_crs, stub=stub, filetype=suffix)

    full_bounds = [gdf.bounds['minx'].min(), gdf.bounds['miny'].min(), gdf.bounds['maxx'].max(), gdf.bounds['maxy'].max()]
    bbox = full_bounds

    if dedup:
        # Just keep most recent lidar for each tile        
        dates = [extract_year(filename) for filename in gdf['filename']]
        gdf['date'] = dates

        bounds = pd.DataFrame(
            gdf.geometry.bounds.values,
            columns=["minx", "miny", "maxx", "maxy"],
            index=gdf.index
        )
        bounds["group"] = cluster_bounds(bounds, tol=0.002)
        gdf_groups = gdf.join(bounds["group"])
        gdf_dedup = (
            gdf_groups.sort_values("date")
            .groupby("group", as_index=False)
            .last()
        )
        gdf_dedup.crs = gdf.crs


        filename_dedup = os.path.join(outdir, 'footprints_unique.gpkg')
        gdf_dedup.to_file(filename_dedup)
        print("Saved:", filename_dedup)
    else:
        gdf_dedup = gdf
        filename_dedup = os.path.join(outdir, f"{stub}_footprints.gpkg")

    base_stub = base_dir.split('/')[-1]
    stub = base_stub + '_' + outdir.split('/')[-1]  # Including the base stub so cropped filenames are unique to avoid parallelization errors.
    mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, stub, outdir, filename_dedup, id_column='filename')  # Deliberately inverting the outdir and tmpdir so the output cropped files go to tmpdir
    ds = merged_ds(mosaic, out_meta, suffix_stub)  # This name shows up in QGIS next to 'Band 1'
    da = ds[suffix_stub].rio.reproject(final_crs)  # This cleans up the nan values around the edge

    parent_dir = os.path.dirname(base_dir) # Best not to save the merged result in the save folder as the original data, in case you want to run the merge again
    outpath = os.path.join(parent_dir, f'{base_stub}_merged{suffix}')

    da.rio.to_raster(outpath, compress="lzw")
    print(f"Saved: {outpath}", flush=True)

    return da


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()

    parser.add_argument('base_dir', help='Directory containing all the tif files to be merged')
    parser.add_argument('--tmpdir', default='/tmp', help='Temporary directory for intermediate files (default: /tmp)')
    parser.add_argument('--suffix', default='.tif', help='Suffix of the files to be merged (default: .tif)')
    parser.add_argument('--subdir', default='', help='Subdirectory inside base_dir containing the files (default: empty, tifs sit directly in base_dir)')
    parser.add_argument('--crs', default=None, help='Force the output to be in a certain EPSG. Need to format the crs in full, e.g. EPSG:3857')
    parser.add_argument("--dont_reproject", action="store_true", help="Don't do any reprojecting. Default: False")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate tiles with the same bbox but different years (use the most recent). Default: False")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    merge_tifs(
        base_dir=args.base_dir,
        tmpdir=args.tmpdir,
        suffix=args.suffix,
        subdir=args.subdir,
        crs=args.crs,
        dont_reproject=args.dont_reproject,
        dedup=args.dedup,
    )
