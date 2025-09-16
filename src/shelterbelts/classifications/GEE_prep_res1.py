# Prep the 1m canopy height file for google earth engine
import glob
import os
import rioxarray as rxr

# +
# # !mkdir /scratch/xe2/cb8590/lidar/DATA_586204/chm_uint8
# -

res1_filenames = glob.glob('/scratch/xe2/cb8590/lidar/DATA_586204/chm/*_res1.tif')
outdir = '/scratch/xe2/cb8590/lidar/DATA_586204/chm_uint8'

# +
# %%time
# Convert all the 1m canopy height models to uint8 to save space
for i, filename in enumerate(res1_filenames):
    da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
    da = da.where(da < 100, 100)  # Truncate trees taller than 100m since we don't have barely any trees that tall in Australia
    da = da.where(da != -9999, 255) # Change nodata to a value compatible with uint8 to save storage space
    da = da.rio.write_nodata(255)
    da = da.astype('uint8')
    outfile = f"{filename.split('/')[-1].split('.')[0]}_uint8.tif"
    outpath = os.path.join(outdir, outfile)
    da.rio.to_raster(outpath, compress="lzw")
    if i%10 == 0:
        print(f"Saved {i}/{len(res1_filenames)}:", outpath)

# Took about an hour. Brought the filesize down from 30MB to 300KB, so 100x smaller wow
# -







# +
# from shelterbelts.apis.worldcover import tif_categorical
# from shelterbelts.classifications.binary_trees import cmap_woody_veg
# outfile = '/Users/christopherbradley/Desktop/GEE_test_booroowa_res1.tif'
# tif_categorical(da, outfile, cmap_woody_veg)
