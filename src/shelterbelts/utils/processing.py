"""Data processing utilities for shelterbelts analysis."""

# Note: I haven't yet tested this or de-duplicated from the original caopy_height file

import os
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
from rasterio.merge import merge
from rasterio.windows import from_bounds
from shapely.geometry import box

from .geo import identify_relevant_tiles_bbox, transform_bbox


def merge_tiles_bbox(bbox, outdir=".", stub="Test", tmpdir='.', footprints_geojson='tiles_global.geojson', id_column='tile', verbose=True):
    """Create a tiff file with just the region of interest. This may use just one tile, or merge multiple tiles.
    
    Parameters
    ----------
    bbox : list
        Bounding box as [minx, miny, maxx, maxy]
    outdir : str
        Output directory
    stub : str
        Filename prefix
    tmpdir : str
        Temporary directory for intermediate files
    footprints_geojson : str
        Filename of the footprints GeoJSON
    id_column : str
        Column name containing tile IDs
    verbose : bool
        Print progress messages
    
    Returns
    -------
    tuple
        (mosaic, out_meta) containing the merged raster and metadata
    """
    os.makedirs(outdir, exist_ok=True)
    canopy_height_dir = tmpdir
    relevant_tiles = identify_relevant_tiles_bbox(bbox, canopy_height_dir, footprints_geojson, id_column)
    footprints_crs = rioxarray.rasterio.rasterio_crs(
        rxr.open_rasterio(os.path.join(canopy_height_dir, footprints_geojson))
    )

    new_relevant_tiles = []
    cropped_tif_filenames = []
    for i, tile in enumerate(relevant_tiles):
        if (i % 100 == 0) and verbose:
            print(f"Working on {i}/{len(relevant_tiles)}: {tile}", flush=True)

        original_tilename = tile
        if tile.endswith('.tif'):
            tile = tile.strip('.tif')
        tiff_file = os.path.join(canopy_height_dir, f"{tile}.tif")

        # Get intersection of the tiff file and the region of interest.
        with rasterio.open(tiff_file) as src:
            # Get bounds of the TIFF file
            tiff_bounds = src.bounds
            tiff_crs = src.crs

            bbox_transformed = transform_bbox(bbox, inputEPSG=footprints_crs.to_string() if hasattr(footprints_crs, 'to_string') else str(footprints_crs), outputEPSG=tiff_crs.to_string() if hasattr(tiff_crs, 'to_string') else str(tiff_crs))  
            roi_box = box(*bbox_transformed)
            intersection_bounds = box(*tiff_bounds).intersection(roi_box).bounds

            # If there is no intersection then don't save a cropped image
            if all(np.isnan(x) for x in intersection_bounds):
                continue
            
            window = from_bounds(*intersection_bounds, transform=src.transform)
            
            # Read data within the window
            out_image = src.read(window=window)
            
            # Skip if intersection is too small
            if out_image.size == 0 or out_image.shape[1] == 0 or out_image.shape[2] == 0:
                continue
    
            out_transform = src.window_transform(window)
            out_meta = src.meta.copy()
            
            new_relevant_tiles.append(original_tilename)

        # Save cropped image
        cropped_tiff_filename = os.path.join(outdir, f"{stub}_{tile}_cropped.tif")
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

        with rasterio.open(cropped_tiff_filename, "w", **out_meta) as dest:
            dest.write(out_image)
        
        # Save this cropped_tif_filename to a list to be deleted later
        cropped_tif_filenames.append(cropped_tiff_filename)

    # Merge the cropped tiffs
    src_files_to_mosaic = []
    for tile in new_relevant_tiles:
        if tile.endswith('.tif'):
            tile = tile.strip('.tif')
        tiff_file = os.path.join(outdir, f'{stub}_{tile}_cropped.tif')        
        src = rasterio.open(tiff_file)
        src_files_to_mosaic.append(src)
    
    # Merge
    if verbose:
        print(f"Merging {len(src_files_to_mosaic)} tiles")
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    for src in src_files_to_mosaic:
        src.close()

    # Remove cropped files
    for filename in cropped_tif_filenames:
        if os.path.exists(filename):
            os.remove(filename)    

    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    return mosaic, out_meta


def merged_ds(mosaic, out_meta, layer_name='canopy_height'):
    """Create an xr.DataArray from the outputs of merge_tiles_bbox.
    
    Parameters
    ----------
    mosaic : np.ndarray
        Mosaic array from merge_tiles_bbox
    out_meta : dict
        Metadata from merge_tiles_bbox
    layer_name : str
        Name for the data variable
    
    Returns
    -------
    xarray.Dataset
        Dataset with the mosaic data
    """
    transform = out_meta['transform']
    height, width = mosaic.shape[1:]
    x = (np.arange(width) + 0.5) * transform.a + transform.c
    y = (np.arange(height) + 0.5) * transform.e + transform.f
    
    coords = {"longitude": x, "latitude": y}
    da = xr.DataArray(
        mosaic[0],
        dims=("latitude", "longitude"),
        coords=coords,
        name=layer_name
    ).rio.write_crs(out_meta['crs'])
    ds = da.to_dataset()
    return ds
