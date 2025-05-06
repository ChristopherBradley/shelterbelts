# Install dependencies
# !pip install rioxarray 

# !python --version

# !pip list

# Load the sentinel imagery and fractional cover
import pickle
with open('../data/ESDALE_3km_2025_ds2i.pkl', 'rb') as file:
    ds = pickle.load(file)

# See all the variables inside the xarray
ds

# Plot the bare ground
ds['bg'].sel(time='2025-01-01').plot()

# Load the canopy height
import rioxarray as rxr
da = rxr.open_rasterio('ESDALE_3km_canopy_height.tif').isel(band=0).drop_vars('band')

# Reproject the canopy height to match the sentinel imagery
from rasterio.enums import Resampling
da_10m = da.rio.reproject_match(ds, resampling=Resampling.max)

# Add the canopy height to the sentinel xarray
ds['canopy_height'] = da_10m

# Plot the canopy height
ds['canopy_height'].plot()

# Save the resampled canopy height as a new tif
ds['canopy_height'].rio.to_raster("ESDALE_3km_canopy_height_10m.tif")
