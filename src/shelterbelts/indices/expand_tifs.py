import os
from pathlib import Path

import rioxarray as rxr
from shapely.geometry import box
import geopandas as gpd

from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds


def expand_tif(filename, folder, outdir, tmpdir='/scratch/xe2/cb8590/tmp', num_pixels=20, pixel_size=10):
    """Expand the tif by a certain number of pixels, to avoid edge effects when running indices at scale
    
    Parameters
    ----------
        filename: A tif file to be expanded.
        folder: A folder of wall-to-wall tif files in the same region surrounding the tif to be expanded
        num_pixels: Number of pixels to be expanded on each edge
        pixel_size: Number of metres per pixel. 
            - This function assumes the crs is EPSG:3857. I haven't added this as a parameter, because I haven't tested if it works in other EPSG's

    Returns
    -------
        ds: an xarray with a band 'expanded'

    Downloads
    ---------
        expanded.tif: A tif file the same as filename but a little bigger   
    
    """
    da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
    minx, miny, maxx, maxy = da.rio.bounds()
    buffer = num_pixels * pixel_size
    expanded_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    # gpd.GeoDataFrame({'geometry': [box(*expanded_bounds)]}, crs='EPSG:3857').to_file('/scratch/xe2/cb8590/tmp/expanded_bounds.gpkg')  # For visualising the expanded bounds in QGIS

    gpkg = f'{Path(merged_folder).parent.stem}_{Path(merged_folder).stem}_footprints.gpkg' # I think this is cleaner than the way I wrote it in bounding_boxes.py, but should give the same result
    if not os.path.exists(gpkg):
        bounding_boxes(merged_folder, crs='EPSG:3857')
    stub = Path(filename).stem
    mosaic, out_meta = merge_tiles_bbox(expanded_bounds, tmpdir, stub, folder, gpkg, 'filename', verbose=False)
    ds_expanded = merged_ds(mosaic, out_meta, 'expanded')

    outpath = os.path.join(outdir, f'{stub}_expanded{num_pixels}.tif')
    ds_expanded['expanded'].rio.to_raster(outpath)
    print(f'Saved: {outpath}')
    
    return ds_expanded


filename = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_148/34_93-148_90_y2024_predicted.tif'
folder = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'

ds = expand_tif(filename, folder, '/scratch/xe2/cb8590/tmp')



Path('/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_148/34_93-148_90_y2024_predicted.tif').stem


filename = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_148/34_93-148_90_y2024_predicted.tif'

da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

da

bounds = da.rio.bounds()

# +
buffer = 100  # 10m x 10 pixels
minx, miny, maxx, maxy = bounds
expanded_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

# Create a shapely box (polygon)
geom = box(*expanded_bounds)

# Save as gpkg
gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:3857').to_file('/scratch/xe2/cb8590/tmp/expanded_bounds.gpkg')

# -



merged_folder = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'

f'{Path(merged_folder).parent.stem}_{Path(merged_folder).stem}_footprints.gpkg'

gpkg = 'subfolders_merged_predicted_footprints.gpkg'

# %%time
gdf = bounding_boxes(merged_folder, crs='EPSG:3857')

gdf = gpd.read_file('/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted/subfolders_merged_predicted_footprints.gpkg')

# +

unique_stub = 'TEST'  # Needs to be unique to this tile, or you get merge errors
# -

mosaic, out_meta = merge_tiles_bbox(expanded_bounds, tmpdir, unique_stub, merged_folder, gpkg, 'filename', verbose=False)     # Need to include the DATA... in the stub so we don't get rasterio merge conflicts
ds_expanded = merged_ds(mosaic, out_meta, 'expanded')

ds_expanded['expanded'].rio.to_raster('/scratch/xe2/cb8590/tmp/expanded.tif')
