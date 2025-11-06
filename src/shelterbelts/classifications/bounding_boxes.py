# +
import os
import argparse
import glob

from shapely.geometry import box
from pyproj import Transformer

import geopandas as gpd
import rioxarray as rxr
import numpy as np
import xarray as xr



def bounding_boxes(folder, outdir=None, stub=None, size_threshold=80, tif_cover_threshold=None, pixel_cover_threshold=None, remove=False, filetype='.tif', crs=None, save_centroids=False, limit=None):
    """Create a geopackage of tif bboxs and remove tifs that don't meet the size or cover threshold
    
    Parameters
    ----------
        folder: Folder containing lots of tifs that we want to extract the bounding box from
        outdir: The output directory to save the results 
        stub: Prefix for output file 
        size_threshold: The number of pixels wide and long the tif should be
        tif_cover_threshold: The minimum percentage cover for tree or no tree pixels that the tif needs to have
        pixel_cover_threshold: The threshold to convert percent_cover pixels into binary pixels
            - Only applies if it isn't already a binary tif
        remove: Whether to actually remove files that don't meet the criteria (otherwise just downloads the gpkg)
        crs: The resulting crs of the gpkg. Originally the default was EPSG:4326, but currently the default is a sample tif inside the folder.
            - EPSG:4326 is what's expected by the sentinel_download scripts

    Downloads
    ---------
        footprint_gpkg: A gpkg with the bounding box of each tif file and corresponding filename
        centroid_gpkg: A gpkg of the centroid of each tif file (this can be easier to view when there are lots of small tif files and you're zoomed out)
    
    """

    if outdir is None:
        outdir = folder
    if stub is None:
        # stub = folder.split('/')[-1].split('.')[0]  # The filename without the path or the suffix
        stub = '_'.join(folder.split('/')[-2:]).split('.')[0]  # The filename and one folder above
        
    veg_tifs = glob.glob(os.path.join(folder, f"*{filetype}"))
    veg_tifs = [f for f in veg_tifs if not os.path.isdir(f)]  # The filetype should handle this, but just in case

    if limit is not None:
        veg_tifs = veg_tifs[:limit]
    
    # Choose the crs for the overall gdf
    if crs is None:
        da = rxr.open_rasterio(veg_tifs[len(veg_tifs)//2]).isel(band=0).drop_vars("band")  # Using the center tiles crs in an attempt to be the most representative
        if da.rio.crs is None:
            da = da.rio.write_crs('EPSG:28355')
        crs = da.rio.crs

    # Create a geopackage of the attributes of each tif
    records = []
    for i, veg_tif in enumerate(veg_tifs):
        if i%10 == 0:
            print(f'Working on {i}/{len(veg_tifs)}: {veg_tif}', flush=True)
        da = rxr.open_rasterio(veg_tif).isel(band=0).drop_vars("band")
        original_crs = str(da.rio.crs)

        height, width = da.shape
        bounds = da.rio.bounds()  
        minx, miny, maxx, maxy = bounds
        if da.rio.crs is None:
            da = da.rio.write_crs('EPSG:28355') # ACT 2015 tifs are missing the crs
        elif da.rio.crs != crs:
            # Trying to just reproject the bounds instead of the whole raster to speed things up.
            transformer = Transformer.from_crs(da.rio.crs, crs, always_xy=True)
            xs, ys = transformer.transform(
                [minx, maxx, maxx, minx],
                [miny, miny, maxy, maxy]
            )
            minx, miny, maxx, maxy = min(xs), min(ys), max(xs),  max(ys)
        # da = da.rio.reproject(crs)
        # height, width = da.shape
        # bounds = da.rio.bounds()  # (minx, miny, maxx, maxy)
        # minx, miny, maxx, maxy = bounds
            
        if pixel_cover_threshold:
            da = (da > pixel_cover_threshold).astype('uint8')

        # year = veg_tif.split('-')[0][-4:] # I don't think there's a generalisable way to figure out the year per tile, since the filenames are all bespoke formats
        
        rec = {
            "filename": os.path.basename(veg_tif),
            "height": height,
            "width": width,
            "crs":original_crs,
            "geometry": box(minx, miny, maxx, maxy),
            # "year":year,  
        }
        
        if tif_cover_threshold:
            unique, counts = np.unique(da.values, return_counts=True)
            category_counts = dict(zip(unique.tolist(), counts.tolist()))
            rec["pixels_0"] = category_counts.get(0, 0)
            rec["pixels_1"] = category_counts.get(1, 0)
        records.append(rec)
        
    # gdf = gpd.GeoDataFrame(records, crs=da.rio.crs)
    gdf = gpd.GeoDataFrame(records, crs=crs)
    
    # Calculate which tifs don't meet our thresholds
    gdf['bad_tif'] = (gdf['height'] < size_threshold) | (gdf['width'] < size_threshold)
    if tif_cover_threshold is not None:
        gdf['percent_trees'] = 100 * gdf['pixels_1'] / (gdf['pixels_1'] + gdf['pixels_0']) 
        gdf['bad_tif'] = gdf['bad_tif'] | (gdf['percent_trees'] > 100 - tif_cover_threshold) | (gdf['percent_trees'] < tif_cover_threshold) # Need to double check I have the < > the right way around

    # Save geopackages
    footprint_gpkg = f"{outdir}/{stub}_footprints.gpkg"
    centroid_gpkg = f"{outdir}/{stub}_centroids.gpkg"
        
    if os.path.exists(footprint_gpkg):  # Odd error where sometimes it's fine to override the file and sometimes it isn't?
        os.remove(footprint_gpkg)

    gdf.to_file(footprint_gpkg)
    print("Saved:", footprint_gpkg)

    if save_centroids:
        # The centroids are easier to view if you zoom out a lot, hence I like saving both the bounding boxes and centroids
        gdf2 = gdf.copy()
        gdf2["geometry"] = gdf2.to_crs("EPSG:6933").centroid.to_crs(gdf2.crs)  # Removing the centroid inaccurate warning
        gdf2.to_file(centroid_gpkg)
        print("Saved:", centroid_gpkg)

    if remove:
        # Remove tifs that are too small, or not enough variation in trees vs no trees
        bad_filenames = gdf.loc[gdf['bad_tif'], 'filename']
        for filename in bad_filenames:
            filepath = os.path.join(outdir, filename)
            os.remove(filepath)

    return gdf




# +
def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('folder', type=str, help='Folder containing lots of tifs that we want to extract the bounding box from')
    parser.add_argument('--outdir', type=str, default=None, help='The output directory to save the results. By default it gets saved in the same directory as the tifs.')
    parser.add_argument('--stub', type=str, default=None, help='Prefix for output file. By default it gets the same name as the folder.')
    parser.add_argument('--size_threshold', type=int, default=80, help='The number of pixels wide and long the tif should be.')
    parser.add_argument('--tif_cover_threshold', type=int, default=10, help='The minimum percentage cover for tree or no tree pixels that the tif needs to have.')
    parser.add_argument('--pixel_cover_threshold', type=int, default=None, help="The threshold to convert percent_cover pixels into binary pixels. Doesn't apply to tifs that are already binary.")
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
        args.pixel_cover_threshold, 
        args.remove, 
        args.filetype,
        args.crs)

# +
# # %%time
# gdf = bounding_boxes('/g/data/xe2/cb8590/Nick_Aus_treecover_10m', filetype='.tiff')
# # gdf = bounding_boxes("/scratch/xe2/cb8590/Worldcover_Australia")

# +
# # # %%time
# # folder = "/scratch/xe2/cb8590/Worldcover_Australia"
# # stub = "Worldcover_Australia"
# # outdir = "/scratch/xe2/cb8590/tmp"
# # filetype = 'tif'
# # crs = None
# # pixel_cover_threshold = None
# # tif_cover_threshold = None  # Takes 10 secs so long as this is None
# # size_threshold = 80
# # remove = False

# # bounding_boxes(folder)

# # Footprints currently aren't working with the .asc files, but centroids are for some reason.
# folder = '/g/data/xe2/cb8590/NSW_5m_DEMs'
# stub = 'NSW_5m_DEMs'
# outdir = "/g/data/xe2/cb8590/Outlines"

# -

# # %%time
# bounding_boxes(filepath, outdir, stub, filetype='.asc', limit=10)

# +
# # # %%time
# gdf = bounding_boxes(filepath, outdir, stub, filetype='.asc', crs='EPSG:4326', limit=10)
# gdf.crs

# +
# size_threshold=80
# tif_cover_threshold=None
# pixel_cover_threshold=None
# remove=False
# filetype='.asc'
# crs=None
# save_centroids=False
# limit=None
