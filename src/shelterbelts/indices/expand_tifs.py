import os
from pathlib import Path
import glob

import rioxarray as rxr
from shapely.geometry import box
import geopandas as gpd

from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds


def expand_tif(filename, folder_merged, outdir, tmpdir='/scratch/xe2/cb8590/tmp', num_pixels=20, pixel_size=10):
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
    mosaic, out_meta = merge_tiles_bbox(expanded_bounds, tmpdir, stub, folder_merged, gpkg, 'filename', verbose=False)
    ds_expanded = merged_ds(mosaic, out_meta, 'expanded')

    outpath = os.path.join(outdir, f'{stub}_expanded{num_pixels}.tif')
    ds_expanded['expanded'].rio.to_raster(outpath)
    print(f'Saved: {outpath}')
    
    return ds_expanded


def expand_tifs(folder_to_expand, folder_merged, outdir, suffix='', non_suffixes=[''], non_contains=[''], limit=None):
    """Run expand_tif on all subfolders in folder_to_expand, preserving the folder structure when writing to outdir"""
    folders = glob.glob(f'{folder_to_expand}/*')

    # I should have better folder management so I don't need to jump through all these hurdles
    folders = [f for f in folders if f.endswith(suffix) 
             and os.path.isdir(f)
             and not any(f.endswith(non_suffix) for non_suffix in non_suffixes)
             and not any(non_contain in f for non_contain in non_contains)
            ]

    if limit:
        folders = folders[:limit]
    for folder in folders:
        filenames = glob.glob(f'{folder}/*')
        sub_outdir = os.path.join(outdir, Path(folder).stem)
        os.makedirs(sub_outdir, exist_ok=True)

        if limit:
            filenames = filenames[:limit]
        for filename in filenames:
            expand_tif(filename, folder_merged, sub_outdir)


filename = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_148/34_93-148_90_y2024_predicted.tif'
folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'
outdir = '/scratch/xe2/cb8590/tmp'
ds = expand_tif(filename, folder_merged, outdir)

folder_to_expand = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders'
folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'
outdir = '/scratch/xe2/cb8590/barra_trees_s4_2024/expanded'

suffix=''
non_suffixes=['_confidence50', '_confidence50_fixedlabels', '_corebugfix', '.tif']
non_contains = ['linear_tifs', 'merged_predicted']

# %%time
expand_tifs(folder_to_expand, folder_merged, outdir, suffix, non_suffixes, non_contains)


