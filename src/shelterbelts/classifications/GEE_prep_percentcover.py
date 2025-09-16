# +
# Prep the 1m canopy height file for google earth engine
import glob
import os
import numpy as np

import rioxarray as rxr

# +
# # !mkdir /scratch/xe2/cb8590/lidar/DATA_586204/treepercent_uint8

# +
# # !ls /scratch/xe2/cb8590/lidar/DATA_586204/chm/
# -

percentcover_filenames = glob.glob('/scratch/xe2/cb8590/lidar/DATA_586204/chm/*_percentcover_res10_height2m.tif')
outdir = '/scratch/xe2/cb8590/lidar/DATA_586204/treepercent_uint8'

filename = percentcover_filenames[0]

# The nodata pixels seem to consistently be in water bodies, so I think it's safe to make these 0.
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
da = da.where(da < 100, 100)  # Can't have more than 100% tree cover
da = da.where(da != -9999, 255) # I don't think there should be any nodata values, but being consistent with the CHM
da = da.rio.write_nodata(255)
da = da.astype('uint8')

outfile = f"{filename.split('/')[-1].split('.')[0]}_uint8.tif"
outpath = os.path.join(outdir, outfile)
da.rio.to_raster(outpath, compress="lzw")
print(f"Saved:", outpath)

# +
# %%time
# Convert all the 10m percent treecover to uint8 to save space
for i, filename in enumerate(percentcover_filenames):
    da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
    da = da.where(da < 100, 100)  # Can't have more than 100% tree cover
    da = da.where(da != -9999, 0) # The nodata pixels seem to consistently be in water bodies, so I think it's safe to make these 0.
    da = da.rio.write_nodata(255)
    da = da.astype('uint8')
    outfile = f"{filename.split('/')[-1].split('.')[0]}_uint8.tif"
    outpath = os.path.join(outdir, outfile)
    da.rio.to_raster(outpath, compress="lzw")
    if i%100 == 0:
        print(f"Saved {i}/{len(percentcover_filenames)}:", outpath)

# Took 2 mins. Brings the filesize down from 40k to 8k, so 5x smaller
# +
# # # !mkdir /scratch/xe2/cb8590/lidar/merged_tifs

# # # %%time
# from shelterbelts.classifications.bounding_boxes import bounding_boxes
# gdf = bounding_boxes(outdir)
# ##  Took about 1 min for 1000 tiles (10m each)
# -

from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds
import geopandas as gpd
import os


filename_bbox = '/scratch/xe2/cb8590/lidar/DATA_586204/treepercent_uint8/r1_c1.geojson'
gdf_bbox = gpd.read_file(filename_bbox)
bbox = gdf_bbox.loc[0, 'geometry'].bounds

filename_footprints = '/scratch/xe2/cb8590/lidar/DATA_586204/treepercent_uint8/treepercent_uint8_footprints.gpkg'
gdf = gpd.read_file(filename_footprints)

dates = [filename.split('-')[0][-6:] for filename in gdf['filename']]
gdf['date'] = dates

# +
# Just keep most recent lidar for each tile
bounds = pd.DataFrame(
    gdf.geometry.bounds.values,
    columns=["minx", "miny", "maxx", "maxy"],
    index=gdf.index
)
def cluster_bounds(bounds, tol=0.02):
    groups = -np.ones(len(bounds), dtype=int)
    group_id = 0
    for i in range(len(bounds)):
        if groups[i] != -1:
            continue  # already assigned
        # compare all remaining rows with this one
        diffs = np.abs(bounds - bounds.iloc[i])
        mask = (diffs <= tol).all(axis=1)
        groups[mask] = group_id
        group_id += 1
    return groups

bounds["group"] = cluster_bounds(bounds, tol=0.002)
gdf_groups = gdf.join(bounds["group"])
gdf_dedup = (
    gdf_groups.sort_values("date")
    .groupby("group", as_index=False)
    .last()
)
gdf_dedup.crs = gdf.crs
filename_dedup = '/scratch/xe2/cb8590/lidar/DATA_586204/treepercent_uint8/treepercent_uint8_footprints_unique_002.gpkg'
gdf_dedup.to_file(filename_dedup)
print(filename_dedup)

# +
# %%time
outdir = '/scratch/xe2/cb8590/tmp'
stub = 'DATA_586204'
tmpdir = '/scratch/xe2/cb8590/lidar/DATA_586204/treepercent_uint8'
# footprints_geojson = os.path.join(tmpdir, 'treepercent_uint8_footprints.gpkg')
footprints_geojson = filename_dedup
mosaic, out_meta = merge_tiles_bbox(bbox, outdir, stub, tmpdir, footprints_geojson, id_column='filename')
ds = merged_ds(mosaic, out_meta, 'percent_cover')

da = ds['percent_cover'].rio.reproject("EPSG:7856")  # GDA2020. Somehow this reprojecting auto-cleans up the nan values on the edge

outpath = '/scratch/xe2/cb8590/lidar/merged_tifs/DATA_586204_percentcover_10m_gda2020_latest.tif'
da.rio.to_raster(outpath, compress="lzw")  # 6MB for the resulting 10m raster
print(outpath)

# Took 3 mins, with 1000 tifs at 10m res 
# -



