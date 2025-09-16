# Prep the 1m canopy height file for google earth engine
import rioxarray as rxr

from shelterbelts.apis.worldcover import tif_categorical
from shelterbelts.classifications.binary_trees import cmap_woody_veg

# !ls ~/Desktop/Boorowa201709-LID1-C3-AHD_6426182_55_0002_0002_chm_res1.tif

# +
# We barely have any trees in australia taller than 100m. 
# -





filename = '/Users/christopherbradley/Desktop/Boorowa201709-LID1-C3-AHD_6426182_55_0002_0002_chm_res1.tif'
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
da = da.where(da < 100, 100)  # Truncate trees taller than 100m since we don't have barely any trees that tall in Australia
da = da.where(da != -9999, 255) # Change nodata to a value compatible with uint8 to save storage space
da = da.rio.write_nodata(255)
da = da.astype('uint8')

da.rio.to_raster('/Users/christopherbradley/Desktop/GEE_test3.tif', compress="lzw")

outfile = '/Users/christopherbradley/Desktop/GEE_test_booroowa_res1.tif'
tif_categorical(da, outfile, cmap_woody_veg)
