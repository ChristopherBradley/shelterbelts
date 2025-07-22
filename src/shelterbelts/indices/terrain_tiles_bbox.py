# Same as terrain tiles but using a bbox as input instead of lat, lon, buffer
# This should probably go in the DAESIM_preprocess repo, but I don't want to make changes there right now while we're finalising things.
import os

import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box

from DAESIM_preprocess.terrain_tiles import run_gdalwarp, interpolate_nan, download_dem, create_xarray

def terrain_tiles(geotif, outdir=".", stub="TEST", tmpdir=".", tile_level=14, interpolate=True):
    """Download 10m resolution elevation from terrain_tiles
    
    Parameters
    ----------
        geotif: String of geotif with the bbox region to download the dem for.
        outdir: The directory to save the final cleaned tiff file.
        stub: The name to be prepended to each file download.
        tmpdir: The directory to save the raw uncleaned tiff file.
        tile_level: The zoom level to determine the pixel size in the resulting tif. See documentation link at the top of this file for more info. 
        interpolate: Boolean flag to decide whether to try to fix bad values or not. 
    
    Downloads
    ---------
        A Tiff file of elevation with severe outlier pixels replaced by the nearest neighbour

    """
    # Find the bbox of the geotif in EPSG:4326
    da = rxr.open_rasterio(geotif, masked=True).isel(band=0)
    bbox_geom = box(*da.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=da.rio.crs)
    bbox = bbox_gdf.to_crs("EPSG:4326").total_bounds.tolist()
    
    # Download the raw data from terrain tiles
    filename = os.path.join(tmpdir, f"{stub}_terrain_original.tif")
    run_gdalwarp(bbox, filename, tile_level)

    if interpolate:
        # Fix bad measurements
        dem, meta = interpolate_nan(filename)        
        filename = os.path.join(outdir, f"{stub}_terrain.tif")
        download_dem(dem, meta, filename)
        ds = create_xarray(dem, meta)
    else:
        # We could use rxr.open_rasterio() but the purpose of the interpolate flag is to reduce computational overhead, so I think it's better not to reload the tif here.  
        ds = None
        
    return ds


outdir = '../../../outdir/'
tmpdir = '../../../tmpdir/'
stub = 'g2_26729'
geotif = f"{outdir}{stub}_categorised.tif"

# %%time
terrain_tiles(geotif, outdir, stub, tmpdir)
