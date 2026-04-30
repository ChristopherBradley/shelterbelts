import os
import gc
import argparse
import glob
import rioxarray as rxr
import rasterio

from shelterbelts.utils.visualisation import tif_categorical, visualise_categories


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


def worldcover_trees(input_data, outdir=".", stub=None, savetif=True, plot=True):
    """
    Convert an ESA WorldCover classification tif into a binary tree-cover tif.

    Pixels labelled as Tree cover (class 10) or Shrubland (class 20) in WorldCover
    are kept as trees (1); everything else becomes non-tree (0). The output tif
    has the same resolution and CRS as the input and can be fed directly into
    :func:`shelterbelts.indices.all_indices.indices_tif`.

    Parameters
    ----------
    input_data : str or xarray.DataArray
        Either a file path to a WorldCover GeoTIFF, or a pre-loaded DataArray.
    outdir : str, optional
        Output directory for saving results.
    stub : str, optional
        Prefix for output filenames. If not provided it is derived from input_data
        when a file path is given; required when passing a DataArray.
    savetif : bool, optional
        Whether to save the results as a GeoTIFF.
    plot : bool, optional
        Whether to generate a PNG visualisation.

    Returns
    -------
    xarray.Dataset
        Dataset with a single woody_veg variable (uint8, 0/1).

    Examples
    --------
    >>> from shelterbelts.utils.filepaths import get_filename
    >>> filename = get_filename('g2_26729_worldcover.tif')
    >>> ds = worldcover_trees(filename, savetif=False, plot=False)
    >>> 'woody_veg' in ds.data_vars
    True

    .. plot::

        import matplotlib.pyplot as plt
        import rioxarray as rxr
        from shelterbelts.classifications.binary_trees import worldcover_trees, cmap_woody_veg, labels_woody_veg
        from shelterbelts.apis.worldcover import worldcover_cmap, worldcover_labels
        from shelterbelts.utils.visualisation import _plot_categories_on_axis
        from shelterbelts.utils.filepaths import get_filename

        filename = get_filename('g2_26729_worldcover.tif')
        da_input = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
        ds = worldcover_trees(filename, savetif=False, plot=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
        _plot_categories_on_axis(ax1, da_input, worldcover_cmap, worldcover_labels, 'WorldCover Input', legend_inside=True)
        _plot_categories_on_axis(ax2, ds['woody_veg'], cmap_woody_veg, labels_woody_veg, 'Binary Tree Cover', legend_inside=True)
        plt.tight_layout()

    """
    if isinstance(input_data, str):
        da = rxr.open_rasterio(input_data).isel(band=0).drop_vars('band')
        if stub is None:
            stub = input_data.split('/')[-1].split('.')[0]
    else:
        da = input_data
        if stub is None:
            raise ValueError("stub must be provided when input_data is a DataArray")

    # WorldCover classes 10 (Tree cover) + 20 (Shrubland) both count as "woody".
    da_trees = (da == 10) | (da == 20)
    da_trees = da_trees.astype('uint8')

    if savetif:
        outpath = os.path.join(outdir, f"{stub}_woody_veg.tif")
        tif_categorical(da_trees, outpath, cmap_woody_veg, tiled=True)

        # Add a pyramid for faster viewing in zoomed out views in QGIS (increases the filesize a bit)
        levels = [2, 4, 8, 16, 32, 64]
        with rasterio.open(outpath, "r+") as src:
            src.build_overviews(levels)

    if plot:
        outpath_png = os.path.join(outdir, f"{stub}_woody_veg.png")
        visualise_categories(da_trees, outpath_png, cmap_woody_veg, labels_woody_veg, "Binary Tree Cover")

    ds = da_trees.to_dataset(name='woody_veg')
    return ds


def canopy_height_trees(input_data, outdir=".", stub=None, savetif=True, plot=True):
    """
    Convert a 1m canopy-height tif into a binary 10m tree-cover tif.

    Any pixel with canopy height ≥ 1m becomes a tree (1) and others become
    non-tree (0). Then uses 'max' resampling to coarsen to 10m resolution.

    Parameters
    ----------
    input_data : str or xarray.DataArray
        Either a file path to a 1m canopy-height GeoTIFF, or a pre-loaded DataArray.
    outdir : str, optional
        Output directory for the saved tif.
    stub : str, optional
        Prefix for output filenames. If None, derived from input_data when a
        file path is given; required when passing a DataArray.
    savetif : bool, optional
        Whether to save the results as a GeoTIFF.
    plot : bool, optional
        Whether to generate a PNG visualisation.

    Returns
    -------
    xarray.Dataset
        Dataset with a single woody_veg variable (uint8, 0/1) at 10m resolution.

    Examples
    --------
    >>> from shelterbelts.utils.filepaths import get_filename
    >>> filename = get_filename('milgadara_1kmx1km_CHM_1m.tif')
    >>> ds = canopy_height_trees(filename, savetif=False, plot=False)
    >>> 'woody_veg' in ds.data_vars
    True

    .. plot::

        import matplotlib.pyplot as plt
        import rioxarray as rxr
        from shelterbelts.classifications.binary_trees import canopy_height_trees, cmap_woody_veg, labels_woody_veg
        from shelterbelts.utils.visualisation import _plot_canopy_height_on_axis, _plot_categories_on_axis
        from shelterbelts.utils.filepaths import get_filename

        filename = get_filename('milgadara_1kmx1km_CHM_1m.tif')
        da_input = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
        ds = canopy_height_trees(filename, savetif=False, plot=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
        _plot_canopy_height_on_axis(ax1, da_input, title='Canopy Height Input (m)')
        _plot_categories_on_axis(ax2, ds['woody_veg'], cmap_woody_veg, labels_woody_veg, 'Binary Tree Cover', legend_inside=True)
        plt.tight_layout()

    """
    if isinstance(input_data, str):
        da = rxr.open_rasterio(input_data).isel(band=0).drop_vars('band')
        if stub is None:
            stub = input_data.split('/')[-1].split('.')[0]
    else:
        da = input_data
        if stub is None:
            raise ValueError("stub must be provided when input_data is a DataArray")

    da_trees = (da >= 1)
    da_trees = da_trees.astype('uint8')

    # Coarsen from 1m to 10m, taking the max so any tall pixel marks the cell as treed.
    da_trees = da_trees.rio.reproject(
        da_trees.rio.crs,
        resolution=10,
        resampling=rasterio.enums.Resampling.max,
    )

    if savetif:
        outpath = os.path.join(outdir, f"{stub}_woody_veg.tif")
        tif_categorical(da_trees, outpath, cmap_woody_veg, tiled=True)
        levels = [2, 4, 8, 16, 32, 64]
        with rasterio.open(outpath, "r+") as src:
            src.build_overviews(levels)

    if plot:
        outpath_png = os.path.join(outdir, f"{stub}_woody_veg.png")
        visualise_categories(da_trees, outpath_png, cmap_woody_veg, labels_woody_veg, "Binary Tree Cover")

    ds = da_trees.to_dataset(name='woody_veg')

    # Trying to avoid memory accumulation
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
    """Apply worldcover_trees or canopy_height_trees to every tif in folder."""
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
    parser.add_argument('--outdir', default='.', help='Output directory for saving results (default: current directory)')
    parser.add_argument('--limit', default=None, type=int, help='Limit processing to the first N tifs in the folder')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_tifs(args.folder, args.func_string, args.outdir, args.limit)
