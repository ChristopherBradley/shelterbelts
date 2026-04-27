import os
import gc
import argparse
import glob
import rioxarray as rxr
import rasterio

from shelterbelts.utils.visualisation import tif_categorical


cmap_woody_veg = {
    0: (240, 240, 240),  # Non-trees are white
    1: (0, 100, 0),      # Trees are green
    255: (0, 100, 200),  # Nodata is blue
}
labels_woody_veg = {
    0: "Non-trees",
    1: "Trees",
    255: "Nodata",
}


def worldcover_trees(filename, outdir=".", stub=None, savetif=True, da=None):
    """
    Convert an ESA WorldCover classification tif into a binary tree-cover tif.

    Pixels labelled as Tree cover (class 10) or Shrubland (class 20) in WorldCover
    are kept as trees (1); everything else becomes non-tree (0). The output tif
    has the same resolution and CRS as the input and can be fed directly into
    :func:`shelterbelts.indices.all_indices.indices_tif`.

    Parameters
    ----------
    filename : str
        Path to a WorldCover GeoTIFF. Required unless ``da`` is supplied.
    outdir : str, optional
        Output directory for the saved tif. Default is ``"."``.
    stub : str, optional
        Prefix for output filenames. If None, derived from ``filename``. Default is None.
    savetif : bool, optional
        When True, write a GeoTIFF with the woody-vegetation colour map and
        pyramid overviews. Default is True.
    da : xarray.DataArray, optional
        Pre-loaded DataArray. If supplied, ``filename`` is not read from disk.

    Returns
    -------
    xarray.Dataset
        Dataset with a single ``woody_veg`` variable (uint8, 0/1).

    Notes
    -----
    When ``savetif=True``, writes ``{outdir}/{stub}_woody_veg.tif`` with a
    categorical colour map and overview pyramids for fast display in QGIS.

    Examples
    --------
    >>> from shelterbelts.utils.filepaths import get_filename
    >>> filename = get_filename('g2_26729_worldcover.tif')
    >>> ds = worldcover_trees(filename, outdir='/tmp', stub='example', savetif=False)
    >>> 'woody_veg' in ds.data_vars
    True
    """
    if da is None:
        da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

    # WorldCover classes 10 (Tree cover) + 20 (Shrubland) both count as "woody".
    da_trees = (da == 10) | (da == 20)
    da_trees = da_trees.astype('uint8')

    if savetif:
        if stub is None:
            stub = filename.split('/')[-1].split('.')[0]
        outpath = os.path.join(outdir, f"{stub}_woody_veg.tif")
        tif_categorical(da_trees, outpath, cmap_woody_veg, tiled=True)

        # Pyramid overviews keep zoomed-out views fast in QGIS.
        levels = [2, 4, 8, 16, 32, 64]
        with rasterio.open(outpath, "r+") as src:
            src.build_overviews(levels)

    ds = da_trees.to_dataset(name='woody_veg')
    return ds


def canopy_height_trees(filename, outdir=".", stub=None, savetif=True, da=None):
    """
    Convert a 1m canopy-height tif into a binary 10m tree-cover tif.

    Any pixel with canopy height ≥ 1m becomes a tree (1); shorter pixels become
    non-tree (0). The raster is then coarsened from 1m to 10m using
    :class:`rasterio.enums.Resampling.max` so that a single tall pixel inside a
    10m cell is enough to mark that cell as treed. The 10m output matches the
    resolution expected by the ``indices/`` pipeline.

    Parameters
    ----------
    filename : str
        Path to a 1m canopy-height GeoTIFF. Required unless ``da`` is supplied.
    outdir : str, optional
        Output directory for the saved tif. Default is ``"."``.
    stub : str, optional
        Prefix for output filenames. If None, derived from ``filename``. Default is None.
    savetif : bool, optional
        When True, write a GeoTIFF with the woody-vegetation colour map and
        pyramid overviews. Default is True.
    da : xarray.DataArray, optional
        Pre-loaded DataArray. If supplied, ``filename`` is not read from disk.

    Returns
    -------
    xarray.Dataset
        Dataset with a single ``woody_veg`` variable (uint8, 0/1) at 10m resolution.
    """
    if da is None:
        da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

    da_trees = (da >= 1)
    da_trees = da_trees.astype('uint8')

    # Coarsen from 1m to 10m, taking the max so any tall pixel marks the cell as treed.
    da_trees = da_trees.rio.reproject(
        da_trees.rio.crs,
        resolution=10,
        resampling=rasterio.enums.Resampling.max,
    )

    if savetif:
        if stub is None:
            stub = filename.split('/')[-1].split('.')[0]
        outpath = os.path.join(outdir, f"{stub}_woody_veg.tif")
        tif_categorical(da_trees, outpath, cmap_woody_veg, tiled=True)
        levels = [2, 4, 8, 16, 32, 64]
        with rasterio.open(outpath, "r+") as src:
            src.build_overviews(levels)

    ds = da_trees.to_dataset(name='woody_veg')

    # Free memory explicitly — canopy-height tiles are often 1m and large.
    da.close()
    da_trees.close()
    del da, da_trees
    gc.collect()

    return ds


funcs = {
    "worldcover_trees": worldcover_trees,
    "canopy_height_trees": canopy_height_trees,
}


def run_tifs(folder, func_string, outdir, limit=None):
    """Apply ``worldcover_trees`` or ``canopy_height_trees`` to every tif in ``folder``."""
    func = funcs[func_string]
    tif_files = glob.glob(os.path.join(folder, "*.tif*"))

    if limit:
        tif_files = tif_files[:limit]

    for i, tif_file in enumerate(tif_files):
        print(f"{i+1}/{len(tif_files)}: Working on {tif_file}")
        func(tif_file, outdir)


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', help='The folder containing all of the input tiffs')
    parser.add_argument('func_string', help="Either 'worldcover_trees' or 'canopy_height_trees'")
    parser.add_argument('--outdir', default='.', help='Output directory for saving results')
    parser.add_argument('--limit', default=None, type=int, help='Limit processing to the first N tifs in the folder')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_tifs(args.folder, args.func_string, args.outdir, args.limit)
