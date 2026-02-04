"""File I/O utilities for shelterbelts analysis."""

import rasterio


def tif_categorical(da, filename="output.tif", colormap=None, tiled=False):
    """Save a GeoTIFF with categorical color scheme.
    
    Parameters
    ----------
    da : xarray.DataArray
        The categorical data to save
    filename : str, optional
        Output filename (default: output.tif)
    colormap : dict, optional
        Color map dictionary mapping values to RGB tuples (0-255)
    tiled : bool, optional
        Whether to use tiled compression (default: False)
    """
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=da.shape[0],
        width=da.shape[1],
        count=1,
        dtype="uint8",
        crs=da.rio.crs,
        transform=da.rio.transform(),
        compress="LZW",
        photometric="palette",
        tiled=tiled,
        nodata=da.rio.nodata
    ) as dst:
        dst.write(da.values, 1)
        if colormap:
            dst.write_colormap(1, colormap)
    
    print(f"Saved: {filename}")
