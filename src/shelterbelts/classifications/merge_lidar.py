# Prep the 1m canopy height file for google earth engine
import glob
import os
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
import numpy as np
import re
from pyproj import CRS as PyprojCRS

from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds


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


def merge_lidar(base_dir, filename_bbox, tmpdir='/scratch/xe2/cb8590/tmp2', suffix='_res1.tif', subdir='chm'):
    """Take a folder of tif files and merge them into a single uint8 raster

    Parameters
    ----------
        base_dir: The directory with all of the tif files to be merged
        filename_bbox: A geojson of the bounding box that was used as an input into ELVIS
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
    suffix_stub = suffix.split('.')[0]
    outdir = os.path.join(base_dir, f'uint8{suffix_stub}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Convert all the files to uint8 to save space. Might be better to do this in the lidar script, so I don't have to re-open each tif.
    glob_path = os.path.join(base_dir, subdir, f'*{suffix}')
    filenames = glob.glob(glob_path)
    # print('glob_path:', glob_path)
    # print('len filenames', len(filenames))
    for i, filename in enumerate(filenames):
        da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
        da = da.where(da < 100, 100)  # Truncate trees taller than 100m since we don't have barely any trees that tall in Australia
        da = da.where(da != -9999, 255) # Change nodata to a value compatible with uint8 to save storage space
        da = da.rio.write_nodata(255)
        da = da.astype('uint8')
        
        # Remove vertical component of the crs
        horizontal_crs = PyprojCRS(da.rio.crs).to_2d()
        da = da.rio.write_crs(horizontal_crs)

        outfile = f"{filename.split('/')[-1].split('.')[0]}_uint8.tif"
        outpath = os.path.join(outdir, outfile)
        da.rio.to_raster(outpath, compress="lzw")
        if i%100 == 0:
            print(f"Saved {i}/{len(filenames)}:", outpath)

    # This gives extra info like number of pixels in each category, but we only care about the filename and geometry
    gdf = bounding_boxes(outdir)

    # This is the bounding box that I used to make the initial request from ELVIS
    gdf_bbox = gpd.read_file(filename_bbox)
    bbox = gdf_bbox.loc[0, 'geometry'].bounds

    # This should work for ACT and NSW naming conventions (just for string ordering, not extracting the exact date)
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
    filename_dedup = os.path.join(outdir, 'footprints_unique_002.gpkg')
    gdf_dedup.to_file(filename_dedup)
    print("Saved:", filename_dedup)

    stub = outdir.split('/')[-1]
    mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, stub, outdir, filename_dedup, id_column='filename')  # I'm deliberately inverting the outdir and tmpdir so cropped files go to tmp
    ds = merged_ds(mosaic, out_meta, 'chm')  # This name doesn't matter, since it's only used in the next step to reproject
    da = ds['chm'].rio.reproject("EPSG:7856")  # GDA2020. This reprojecting also cleans up the nan values on the edge

    outpath = os.path.join(base_dir, f'merged{suffix}')
    da.rio.to_raster(outpath, compress="lzw")  # 200MB for the resulting 1m raster in a 50km x 50km area
    print(f"Saved: {outpath}", flush=True)

    return da
    
    # inpath = '/scratch/xe2/cb8590/lidar/merged_tifs/DATA_586204_chm_1m_gda2020_latest.tif'
    # da = rxr.open_rasterio(outpath)
    # outpath = '/scratch/xe2/cb8590/lidar/merged_tifs/DATA_586204_chm_1m_gda2020_latest_tiled512.tif'
    # da.rio.to_raster(outpath, compress="lzw", blocksize=512)  # 200MB for the resulting 1m raster in a 50km x 50km area
    # # !gdaladdo {outpath} 2 4 8 16 32 64 


# +
import argparse

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', required=True, help='Directory containing all the tif files to be merged')
    parser.add_argument('--filename_bbox', required=True, help='GeoJSON file of the bounding box used as input into ELVIS')
    parser.add_argument('--tmpdir', default='/scratch/xe2/cb8590/tmp', help='Temporary directory for intermediate files (default: /scratch/xe2/cb8590/tmp)')
    parser.add_argument('--suffix', default='_res1.tif', help='Suffix of the files to be merged (default: _res1.tif)')
    parser.add_argument('--subdir', default='chm', help='Subdirectory inside base_dir containing the files (default: chm)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    merge_lidar(
        base_dir=args.base_dir,
        filename_bbox=args.filename_bbox,
        tmpdir=args.tmpdir,
        suffix=args.suffix,
        subdir=args.subdir
    )


# +
# # %%time
# base_dir = '/scratch/xe2/cb8590/lidar/DATA_587060'
# filename_bbox = '/scratch/xe2/cb8590/lidar/polygons/DATA_587060.geojson'
# # filename_bbox = '/scratch/xe2/cb8590/lidar/polygons/r1_c2.geojson'
# merge_lidar(base_dir, filename_bbox, subdir='chm', suffix='_percentcover_res10_height2m.tif')

# # # Took 4 mins