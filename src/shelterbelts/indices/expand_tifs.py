import os
from pathlib import Path
import glob
import argparse

import rioxarray as rxr

from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds


from shelterbelts.utils.filepaths import default_tmpdir

def expand_tif(filename, folder_merged, outdir, gpkg=None, tmpdir=default_tmpdir, num_pixels=20, pixel_size=10):
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
    if not gpkg:
        gpkg = os.path.join(folder_merged, f'{Path(folder_merged).parent.stem}_{Path(folder_merged).stem}_footprints.gpkg') # I think this is cleaner than the way I wrote it in bounding_boxes.py, but should give the same result
    stub = f'{Path(filename).stem}_expanded'
    if not os.path.exists(gpkg):
        # Just do this once before running these in parallel, that way you don't run into parallel issues of multiple jobs trying to create the gpkg at the same time.
        print(f"Footprints gpkg doesn't exist. Please run bounding_boxes.py first! {gpkg}")
        return None
        # print(f"Creating bounding boxes for: {folder_merged}, because of filename: {filename}")
        # bounding_boxes(folder_merged, crs='EPSG:3857')  
    mosaic, out_meta = merge_tiles_bbox(expanded_bounds, tmpdir, stub, folder_merged, gpkg, 'filename', verbose=False)
    ds_expanded = merged_ds(mosaic, out_meta, 'expanded')

    outpath = os.path.join(outdir, f'{stub}{num_pixels}.tif')
    ds_expanded['expanded'].rio.to_raster(outpath)
    print(f'Saved: {outpath}')
    
    return ds_expanded


# +
def expand_tifs(folder_to_expand, folder_merged, outdir, limit=None, gpkg=None):
    """Run expand_tif on all subfolders in folder_to_expand, preserving the folder structure when writing to outdir"""
    filenames = glob.glob(f'{folder_to_expand}/*.tif')
    filenames = [f for f in filenames if not os.path.isdir(f)]  # Remove the uint8_predicted
    filenames = [f for f in filenames if not 'merged' in f]
    sub_outdir = os.path.join(outdir, Path(folder_to_expand).stem)
    os.makedirs(sub_outdir, exist_ok=True)

    if limit:
        filenames = filenames[:limit]
    for filename in filenames:
        expand_tif(filename, folder_merged, sub_outdir, gpkg)

# Takes ~15 mins per folder


# -

def parse_arguments():
    """Parse command line arguments for expand_tifs."""
    parser = argparse.ArgumentParser(description="Expand a collection of TIFs in subfolders to reduce edge effects.")
    parser.add_argument("--folder_to_expand", required=True, help="Root folder containing subfolders of TIF files to expand.")
    parser.add_argument("--folder_merged", required=True, help="Folder containing wall-to-wall merged TIFs for context.")
    parser.add_argument("--outdir", required=True, help="Output directory to save expanded TIFs (folder structure will be preserved).")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of subfolders to process.")
    parser.add_argument("--gpkg", type=str, default=None, help="Provide your own gpkg instead of assuming it from the folder structure.")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    expand_tifs(folder_to_expand=args.folder_to_expand, 
                folder_merged=args.folder_merged, 
                outdir=args.outdir,
                limit=args.limit,
                gpkg=args.gpkg
               )

# +
# filename = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_148/34_93-148_90_y2024_predicted.tif'
# folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'
# outdir = '/scratch/xe2/cb8590/tmp'

# +
# ds = expand_tif(filename, folder_merged, outdir)

# +
# # %%time
# folder_to_expand = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_144'
# folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'
# outdir = '/scratch/xe2/cb8590/barra_trees_s4_2024/expanded'

# +
# # %%time
# expand_tifs(folder_to_expand, folder_merged, outdir, limit=10)
# # 7 secs for 10, means about 30 mins per folder
# -


