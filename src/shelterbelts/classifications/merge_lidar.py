# Prep the 1m canopy height file for google earth engine
import glob
import os
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
import numpy as np
import re
from pyproj import CRS as PyprojCRS
from shapely.geometry import box      


from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds, transform_bbox, identify_relevant_tiles_bbox


def extract_year(name):
    first_number = re.search(r'\d', name).start()
    year = name[first_number:first_number + 6]    # YYYYMM or YYYY__ I'm just using this to sort the year-month, so it doesn't matter if it's formatted perfectly (lots will have an extra 2 digits that can be ignored)
    return year


# Find tiles with bounds that are matching (close enough)
def cluster_bounds(bounds, tol=0.02):
    groups = -np.ones(len(bounds), dtype=int)
    group_id = 0
    for i in range(len(bounds)):
        if groups[i] != -1:
            continue  
        diffs = np.abs(bounds - bounds.iloc[i])
        mask = (diffs <= tol).all(axis=1)
        groups[mask] = group_id
        group_id += 1
    return groups


def merge_lidar(base_dir, tmpdir='/scratch/xe2/cb8590/tmp', suffix='_res1.tif', subdir='chm', crs=None, dont_reproject=False, dedup=True):
    """Take a folder of tif files and merge them into a single uint8 raster

    Parameters
    ----------
        base_dir: The directory with all of the tif files to be merged
        tmpdir: Directory where files can be happily deleted
        suffix: The suffix of the files to be merged
        subdir: The directory inside basedir that contains the files to be merged

    Returns
    -------
        da: xarray.DataArray of merged tifs in base_dir within the bounds of filename_bbox

    Downloads
    ---------
        merged.tif: A geotif of the da
        
    """
    # Convert all the files to uint8 to save space. Might be better to do this in the lidar script, so I don't have to re-open each tif.
    glob_path = os.path.join(base_dir, subdir, f'*{suffix}')
    filenames = glob.glob(glob_path)

    # Use the middle filename to choose a crs. 
    da = rxr.open_rasterio(filenames[len(filenames)//2]).isel(band=0).drop_vars("band")  
    if not crs:
        final_crs = da.rio.estimate_utm_crs()
    else: 
        final_crs = crs
    print(f"Merging with crs: {final_crs}")

    suffix_stub = suffix.split('.')[0]

    if da.dtype == 'uint8':
        outdir = base_dir   # Don't convert to uint8
    else:
        outdir = os.path.join(base_dir, f'uint8{suffix_stub}')

    # Convert to uint8. Should probs add a parameter to not do this, in case we want to save the merged raster in the original datatype
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        for i, filename in enumerate(filenames):
            da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
            da = da.where(da < 100, 100)  # Truncate trees taller than 100m since we don't have barely any trees that tall in Australia
            da = da.where(da != -9999, 255) # Change nodata to a value compatible with uint8 to save storage space
            da = da.rio.write_nodata(255)
            da = da.astype('uint8')
            
            if not dont_reproject:
                da = da.rio.reproject(final_crs) 
    
            outfile = f"{filename.split('/')[-1].split('.')[0]}_uint8.tif"
            outpath = os.path.join(outdir, outfile)
            da.rio.to_raster(outpath, compress="lzw")
            if i%100 == 0:
                print(f"Saved {i}/{len(filenames)}:", outpath)

    # This gives extra info like number of pixels in each category, but we only care about the filename and geometry
    stub = f"{'_'.join(outdir.split('/')[-2:]).split('.')[0]}_{suffix_stub}"  # The filename and one folder above with the suffix. 
    gdf = bounding_boxes(outdir, crs=final_crs, stub=stub)
    
    # This is the bounding box that I used to make the initial request from ELVIS
    full_bounds =[gdf.bounds['minx'].min(), gdf.bounds['miny'].min(), gdf.bounds['maxx'].max(), gdf.bounds['maxy'].max()]
    bbox = full_bounds

    # Visualise the full bounds in QGIS
    # gds_full_bounds = gpd.GeoSeries([box(*full_bounds)], crs=gdf.crs)
    # gds_full_bounds.to_file('/scratch/xe2/cb8590/tmp/DATA_717827_full_bounds.geojson')

    if dedup:
        # This should work for ACT and NSW naming conventions (just for string ordering, not extracting the exact date). 
        # If I already know the tiles don't overlap at all, I should skip this step. 
        dates = [extract_year(filename) for filename in gdf['filename']]
        gdf['date'] = dates
    
        # Just keep most recent lidar for each tile
        bounds = pd.DataFrame(
            gdf.geometry.bounds.values,
            columns=["minx", "miny", "maxx", "maxy"],
            index=gdf.index
        )
        bounds["group"] = cluster_bounds(bounds, tol=0.002)
        gdf_groups = gdf.join(bounds["group"])
        gdf_dedup = (
            gdf_groups.sort_values("date")
            .groupby("group", as_index=False)
            .last()
        )
        gdf_dedup.crs = gdf.crs
            
        filename_dedup = os.path.join(outdir, 'footprints_unique.gpkg')
        gdf_dedup.to_file(filename_dedup)
        print("Saved:", filename_dedup)
    else:
        gdf_dedup = gdf
        filename_dedup = os.path.join(outdir, f"{stub}_footprints.gpkg")

    # Finally merge the relevant tiles
    base_stub = base_dir.split('/')[-1]
    stub = base_stub + '_' + outdir.split('/')[-1]  # Need to include the base stub so cropped filenames are unique, so rasterio doesn't die when submitting multiple jobs at once.
    mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, stub, outdir, filename_dedup, id_column='filename')  # I'm deliberately inverting the outdir and tmpdir so the output cropped files go to tmp
    ds = merged_ds(mosaic, out_meta, suffix_stub)  # This name shows up in QGIS next to 'Band 1'
    da = ds[suffix_stub].rio.reproject(final_crs)  # This reprojecting should clean up the nan values on the edge

    parent_dir = os.path.dirname(base_dir) # Best not to save the merged result in the save folder as the original data, in case you want to run the merge again
    outpath = os.path.join(parent_dir, f'{base_stub}_merged{suffix}')
    
    # Might be nice to try to copy the colour scheme from one of the original tifs and add it to the merged output using rasterio
    da.rio.to_raster(outpath, compress="lzw")  # 200MB for the resulting 1m raster in a 50km x 50km area
    print(f"Saved: {outpath}", flush=True)

    return da
    
    # inpath = '/scratch/xe2/cb8590/lidar/merged_tifs/DATA_586204_chm_1m_gda2020_latest.tif'
    # da = rxr.open_rasterio(outpath)
    # outpath = '/scratch/xe2/cb8590/lidar/merged_tifs/DATA_586204_chm_1m_gda2020_latest_tiled512.tif'
    # da.rio.to_raster(outpath, compress="lzw", blocksize=512)  # 200MB for the resulting 1m raster in a 50km x 50km area
    # # !gdaladdo {outpath} 2 4 8 16 32 64 
    # Seems like earth engine already does the tiling by default, so I shouldn't need to do it myself if that's the main use case.

# +
import argparse

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', required=True, help='Directory containing all the tif files to be merged')
    parser.add_argument('--tmpdir', default='/scratch/xe2/cb8590/tmp', help='Temporary directory for intermediate files (default: /scratch/xe2/cb8590/tmp)')
    parser.add_argument('--suffix', default='_res1.tif', help='Suffix of the files to be merged (default: _res1.tif)')
    parser.add_argument('--subdir', default='chm', help='Subdirectory inside base_dir containing the files (default: chm)')
    parser.add_argument('--crs', default=None, help='Force the output to be in a certain EPSG. Need to format the crs in full, e.g. EPSG:3857')
    parser.add_argument("--dont_reproject", action="store_true", help="Don't do any reprojecting. Default: False")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    merge_lidar(
        base_dir=args.base_dir,
        tmpdir=args.tmpdir,
        suffix=args.suffix,
        subdir=args.subdir,
        crs=args.crs,
        dont_reproject=args.dont_reproject
    )

