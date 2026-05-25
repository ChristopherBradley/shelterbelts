import os
import argparse
import glob

from shapely.geometry import box
from pyproj import Transformer

import geopandas as gpd
import rioxarray as rxr
import numpy as np


def bounding_boxes(folder, outdir=None, stub=None, size_threshold=80, tif_cover_threshold=None, remove=False, filetype='.tif', crs=None, save_centroids=False, limit=None, verbose=True):
    """
    Collect footprints for every GeoTIFF in a folder.

    Parameters
    ----------
    folder : str
        Folder of tifs.
    outdir : str, optional
        Output directory for saving results. Defaults to the same folder.
    stub : str, optional
        Prefix for output filenames. By default it gets derived from the folder name.
    size_threshold : int, optional
        Minimum acceptable tile side length in pixels.
    tif_cover_threshold : int, optional
        Minimum percentage of tree (or non-tree) pixels required. When None
        the cover check is skipped.
    remove : bool, optional
        Delete tifs that don't meet the size or percent cover threshold.
    filetype : str, optional
        Suffix matching the files to scan.
    crs : str or pyproj.CRS, optional
        CRS for the output GeoPackage. If None, estimated from the middle tile
        in the folder.
    save_centroids : bool, optional
        Also write a centroid GeoPackage (useful for very zoomed-out views).
    limit : int, optional
        Only process a certain number of tifs. By default it processes all tifs.
    verbose : bool, optional
        Print progress every 100 tifs.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per tif with columns:
        filename, height, width, crs, geometry
        plus pixels_0/pixels_1/percent_trees, bad_tif (boolean) when tif_cover_threshold is used.

    Notes
    -----
    Writes to disk:

    - {outdir}/{stub}_footprints.gpkg — one polygon per tile
    - {outdir}/{stub}_centroids.gpkg — one point per tile (only if save_centroids=True)

    Examples
    --------
    Collect footprints from the bundled folder of two binary tree tifs:

    >>> gdf = bounding_boxes('data/multiple_binary_tifs', filetype='.tiff', verbose=False) 
    Saved: data/multiple_binary_tifs/data_multiple_binary_tifs_footprints.gpkg
    """
    if outdir is None:
        outdir = folder
    if stub is None:
        stub = '_'.join(folder.split('/')[-2:]).split('.')[0]

    veg_tifs = glob.glob(os.path.join(folder, f"*{filetype}"))
    veg_tifs = [f for f in veg_tifs if not os.path.isdir(f)]

    if len(veg_tifs) == 0:
        print("No files found, perhaps you used the wrong folder?")
        print("Folder:", folder)
        return None

    if limit is not None:
        veg_tifs = veg_tifs[:limit]

    # Pick a representative CRS from the middle tile if the caller didn't specify one.
    if crs is None:
        da = rxr.open_rasterio(veg_tifs[len(veg_tifs)//2]).isel(band=0).drop_vars("band")
        if da.rio.crs is None:
            da = da.rio.write_crs('EPSG:28355')
        crs = da.rio.crs

    records = []
    for i, veg_tif in enumerate(veg_tifs):
        if verbose and i % 100 == 0:
            print(f'Working on {i}/{len(veg_tifs)}: {veg_tif}', flush=True)
        da = rxr.open_rasterio(veg_tif).isel(band=0).drop_vars("band")
        original_crs = str(da.rio.crs)

        height, width = da.shape
        bounds = da.rio.bounds()
        minx, miny, maxx, maxy = bounds
        if da.rio.crs is None:
            da = da.rio.write_crs('EPSG:28355')  # ACT 2015 tifs on ELVIS are missing the crs
        elif da.rio.crs != crs:
            # Reproject the four corners
            transformer = Transformer.from_crs(da.rio.crs, crs, always_xy=True)
            xs, ys = transformer.transform(
                [minx, maxx, maxx, minx],
                [miny, miny, maxy, maxy],
            )
            minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)

        rec = {
            "filename": os.path.basename(veg_tif),
            "height": height,
            "width": width,
            "crs": original_crs,
            "geometry": box(minx, miny, maxx, maxy),
        }

        if tif_cover_threshold:
            unique, counts = np.unique(da.values, return_counts=True)
            category_counts = dict(zip(unique.tolist(), counts.tolist()))
            rec["pixels_0"] = category_counts.get(0, 0)
            rec["pixels_1"] = category_counts.get(1, 0)
        records.append(rec)

    gdf = gpd.GeoDataFrame(records, crs=crs)

    gdf['bad_tif'] = (gdf['height'] < size_threshold) | (gdf['width'] < size_threshold)
    if tif_cover_threshold is not None:
        gdf['percent_trees'] = 100 * gdf['pixels_1'] / (gdf['pixels_1'] + gdf['pixels_0'])
        gdf['bad_tif'] = (
            gdf['bad_tif']
            | (gdf['percent_trees'] > 100 - tif_cover_threshold)
            | (gdf['percent_trees'] < tif_cover_threshold)
        )

    footprint_gpkg = f"{outdir}/{stub}_footprints.gpkg"
    centroid_gpkg = f"{outdir}/{stub}_centroids.gpkg"


    if os.path.exists(footprint_gpkg):  # There's an odd error where it's sometimes fine to overwrite the gpkg, but sometimes not, so removing first to be safe.
        os.remove(footprint_gpkg)

    gdf.to_file(footprint_gpkg)
    print("Saved:", footprint_gpkg)

    if save_centroids:
        gdf2 = gdf.copy()
        gdf2["geometry"] = gdf2.to_crs("EPSG:6933").centroid.to_crs(gdf2.crs)  # This fixes the "centroid inaccurate" warning
        gdf2.to_file(centroid_gpkg)
        print("Saved:", centroid_gpkg)

    if remove:
        bad_filenames = gdf.loc[gdf['bad_tif'], 'filename']
        for filename in bad_filenames:
            filepath = os.path.join(outdir, filename)
            os.remove(filepath)

    return gdf


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str, help='Folder containing lots of tifs that we want to extract the bounding box from')
    parser.add_argument('--outdir', type=str, default=None, help='The output directory to save the results. By default it gets saved in the same directory as the tifs.')
    parser.add_argument('--stub', type=str, default=None, help='Prefix for output file. By default it gets the same name as the folder.')
    parser.add_argument('--size_threshold', type=int, default=80, help='The number of pixels wide and long the tif should be.')
    parser.add_argument('--tif_cover_threshold', type=int, default=10, help='The minimum percentage cover for tree or no tree pixels that the tif needs to have.')
    parser.add_argument('--filetype', type=str, default=".tif", help='Suffix of the tif files. Probably .tif or .tiff')
    parser.add_argument('--remove', action="store_true", help="Whether to actually remove files that don't meet the criteria (otherwise just downloads the gpkg)")
    parser.add_argument('--crs', type=str, default=None, help="The crs of the resulting gpkg. If not provided, then a random tif is chosen and the crs estimated from that.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    bounding_boxes(
        args.folder,
        args.outdir,
        args.stub,
        args.size_threshold,
        args.tif_cover_threshold,
        args.remove,
        args.filetype,
        args.crs,
    )
