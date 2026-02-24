# Utilities for working with tiled raster datasets

import os
import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box

from shelterbelts.utils.visualisation import tif_categorical


# Colourmap constants
gullies_cmap = {
    0: (255, 255, 255),
    1: (0, 0, 255),
}


def identify_relevant_tiles_bbox(bbox=[147.735717, -42.912122, 147.785717, -42.862122], canopy_height_dir=".", footprints_geojson='tiles_global.geojson', id_column='tile'):
    """Find the tiles that overlap with the region of interest
    
    Parameters
    ----------
    bbox : list
        Bounding box as [minx, miny, maxx, maxy]
    canopy_height_dir : str
        Directory containing the footprints_geojson file
    footprints_geojson : str
        Path to GeoJSON file with tile footprints
    id_column : str
        Column name in GeoJSON with tile identifiers
    
    Returns
    -------
    list
        List of tile identifiers that intersect the bbox
    """
    roi_polygon = box(*bbox)
    
    filename = os.path.join(canopy_height_dir, footprints_geojson)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Footprints file not found: {filename}")

    # Load the tiles
    gdf = gpd.read_file(filename)

    # Find any tiles that intersect with this polygon
    intersecting_tiles = gdf[gdf.geometry.intersects(roi_polygon)]
    relevant_tiles = intersecting_tiles[id_column].tolist()
    
    return relevant_tiles


def transform_bbox(
    bbox=[148.464499, -34.394042, 148.474499, -34.384042],
    inputEPSG="EPSG:4326",
    outputEPSG="EPSG:3857"
):
    """Transform bounding box coordinates between CRS systems"""
    transformer = Transformer.from_crs(inputEPSG, outputEPSG, always_xy=True)
    # bbox = (minx, miny, maxx, maxy)
    x1, y1 = transformer.transform(bbox[0], bbox[1])
    x2, y2 = transformer.transform(bbox[2], bbox[3])
    return (x1, y1, x2, y2)


def merge_tiles_bbox(bbox, outdir=".", stub="Test", tmpdir='.', footprints_geojson='tiles_global.geojson', id_column='tile', verbose=True):
    """Create a tiff file with just the region of interest. This may use just one tile, or merge multiple tiles"""

    os.makedirs(outdir, exist_ok=True)
    canopy_height_dir = tmpdir
    relevant_tiles = identify_relevant_tiles_bbox(bbox, canopy_height_dir, footprints_geojson, id_column)
    footprints_crs = gpd.read_file(os.path.join(canopy_height_dir, footprints_geojson)).crs

    new_relevant_tiles = []
    cropped_tif_filenames = []
    for i, tile in enumerate(relevant_tiles):
        if (i % 100 == 0) and verbose:
            print(f"Working on {i}/{len(relevant_tiles)}: {tile}", flush=True)

        original_tilename = tile
        if tile.endswith('.tif'):
            tile = tile.strip('.tif')
        tiff_file = os.path.join(canopy_height_dir, f"{tile}.tif")

        # Get intersection of the tiff file and the region of interest
        with rasterio.open(tiff_file) as src:
            # Get bounds of the TIFF file
            tiff_bounds = src.bounds
            tiff_crs = src.crs

            bbox_transformed = transform_bbox(bbox, inputEPSG=footprints_crs, outputEPSG=tiff_crs)  
            roi_box = box(*bbox_transformed)
            intersection_bounds = box(*tiff_bounds).intersection(roi_box).bounds

            # If there is no intersection then don't save a cropped image, and remove this from the relevant tiles. 
            if all(np.isnan(x) for x in intersection_bounds):
                continue
            
            window = from_bounds(*intersection_bounds, transform=src.transform)
            
            # Read data within the window
            out_image = src.read(window=window)
            
            # Solve potential 0x418 error
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
    
    # This assumes the the crs of all the input geotifs is the same
    if verbose:
        print(f"Merging {len(src_files_to_mosaic)} tiles")
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    for src in src_files_to_mosaic:
        src.close()

    # Remove the cropped_tif_filenames so they don't clog up the tmpdir
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
    """Create an xr.DataArray from the outputs of merge_tiles_bbox"""
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


def crop_and_rasterize(geotif, feature_gdb, outdir=".", stub="TEST", save_gpkg=True, savetif=True, layer='HydroLines', feature_name=None):
    """Crop vector features to the region of interest and rasterize them.
    
    This function generalizes rasterization of any vector features (hydrolines, roads, etc.)
    to the bounding box of a reference raster.
    
    Parameters
    ----------
    geotif : str or xarray.DataArray
        Path to a raster file (GeoTIFF, etc.) or an xarray DataArray with geospatial metadata.
        Used to determine the bounding box and CRS for cropping vector features.
    feature_gdb : str
        Path to GDB or GPKG file containing the vector features
    outdir : str, default='.'
        Output directory to save results
    stub : str, default='TEST'
        Prefix for output files
    save_gpkg : bool, default=True
        Whether to save cropped features as GPKG
    savetif : bool, default=True
        Whether to save rasterized output as GeoTIFF
    layer : str, default='HydroLines'
        Layer name within the GDB (e.g., 'HydroLines', 'NationalRoads_2025_09')
    feature_name : str, optional
        Name for the rasterized feature (e.g., 'gullies', 'roads'). 
        Defaults to layer name converted to lowercase

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Geodataframe of the features in the region of interest
    ds : xarray.Dataset
        Dataset with rasterized feature layer

    """
    if feature_name is None:
        feature_name = layer.lower()
    
    # Use raster to get bounding box and CRS
    if isinstance(geotif, str):
        da = rxr.open_rasterio(geotif, masked=True).isel(band=0)
    else:
        da = geotif
    raster_bounds = da.rio.bounds()
    raster_crs = da.rio.crs

    # Reproject raster bounding box to feature CRS (more computationally efficient than the other way around)
    bbox_geom = box(*raster_bounds)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=raster_crs)
    bbox_gdf = bbox_gdf.to_crs('EPSG:4283')  # Standard GDB CRS

    if feature_gdb.endswith('.gpkg'):
        gdf = gpd.read_file(feature_gdb)  # pre-cropped geopackage
    else: 
        # This file is about 2GB, but can be spatially indexed so loads really fast
        gdf = gpd.read_file(feature_gdb, layer=layer, bbox=bbox_gdf)

    if save_gpkg:
        cropped_path = os.path.join(outdir, f"{stub}_{layer}_cropped.gpkg")
        gdf.to_file(cropped_path)
        print("Saved:", cropped_path)

    gdf = gdf.to_crs(da.rio.crs)
    shapes = [(geom, 1) for geom in gdf.geometry]
    transform = da.rio.transform()
    if not shapes:
        rasterized_feature = np.zeros(da.shape, dtype=np.uint8)
    else:
        rasterized_feature = rasterize(
            shapes,
            out_shape=da.shape,
            transform=transform,
            fill=0
        )
    ds = da.to_dataset(name='input')
    ds[feature_name] = (["y", "x"], rasterized_feature)

    if savetif:
        filename_feature = os.path.join(outdir, f"{stub}_{layer}.tif")
        tif_categorical(ds[feature_name], filename_feature, colormap=gullies_cmap)

    return gdf, ds
