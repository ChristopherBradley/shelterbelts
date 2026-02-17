# %% [markdown]
# # BARRA API Demo

# %%
from shelterbelts.apis.barra_daily import barra_daily

# %% [markdown]
# ## Default Parameters
# The default location is a ~2km x 2km region near Milgadara, NSW, Australia for 2020-2021. This should take about 20 secs to load.

# %%
# %%time
ds = barra_daily()
ds

# %% [markdown]
# ## Visualising the Results
# The default output includes a wind rose visualisation showing wind speed distribution by direction.

# %%
from shelterbelts.apis.barra_daily import wind_rose

wind_rose(ds)

# %% [markdown]
# ## Wind Statistics
# You can also get the dominant wind direction above a certain threshold and wind statistics.

# %%
from shelterbelts.apis.barra_daily import dominant_wind_direction, wind_dataframe

direction, df_counts = dominant_wind_direction(ds, threshold_kmh=15)
print(f"Dominant wind direction (>15 km/h): {direction}")
print(df_counts)

# %%
df_wind, max_speed, max_direction = wind_dataframe(ds)
print("Wind speed and direction frequency (%):")
print(df_wind)
print(f"Maximum wind speed: {max_speed} km/h from {max_direction}")

# %% [markdown]
# ## Changing Variables
# You can download different BARRA variables. The default is Eastward and Northward Near-Surface Wind, which can be combined to get wind direction and magnitude. See the full documentation for other available variables: https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f2551_3726_7908_8861

# %%
barra_daily(variables=["sfcWind"], plot=False) # Near surface wind speed

# %% [markdown]
# ## Changing Time Period
# Adjust the start_year and end_year parameters to download data for different time periods. These years are inclusive, so using the same value gets one year of data.

# %%
# %%time
ds_longer = barra_daily(start_year="2020", end_year="2020", save_netcdf=False, plot=False)
wind_rose(ds_longer)

# %% [markdown]
# ## Changing Location
# Use the lat and lon arguments to download data for a different location. Here is an example of ANU in Canberra.

# %%
ds_canberra = barra_daily(lat=-35.287, lon=149.117)
wind_rose(ds_canberra)

# %% [markdown]
# ## Changing Output Directory
# Specify a custom output directory for the saved files.

# %%
import os
outdir = "outdir"
os.makedirs(outdir, exist_ok=True)
ds = barra_daily(outdir=outdir)

# %% [markdown]
# ## Changing Filename Prefix
# Choose a stub to be used as the prefix for output filenames.

# %%
ds = barra_daily(stub="DEMO")

# %% [markdown]
# ## Disabling NetCDF Output
# Set `save_netcdf=False` to skip saving the NetCDF file. 
# Useful when running within a larger pipeline.

# %%
ds = barra_daily(save_netcdf=False)

# %% [markdown]
# ## Disabling Visualization
# Set `plot=False` to skip generating and saving the wind rose PNG visualisation. 
# This can be used in combination with `save_netcdf=False` to avoid any outputs.

# %%
ds = barra_daily(plot=False)

# %% [markdown]
# ## Using Different Temporal Resolutions
# You can request data at different temporal resolutions: '20min', '1hr', 'day', or 'mon'.
# Monthly data is much faster to download.

# %%
# %%time
ds_monthly = barra_daily(temporal='mon', save_netcdf=False, plot=False)
print(f"Monthly data shape: {ds_monthly.dims}")
print(f"Daily data shape: {ds.dims}")

# %% [markdown]
# ## Command Line Interface
# You can also use the function from the command line with the same defaults and parameters.

# %%
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

# %%
# !python shelterbelts/apis/barra_daily.py --help

# %%
# %%time
# !python shelterbelts/apis/barra_daily.py

# %%
# !python shelterbelts/apis/barra_daily.py --lat -35.287 --lon 149.117 --start_year 2020 --end_year 2020 --stub command_line --outdir ../notebooks/apis/outdir

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# !rm ../notebooks/apis/*.nc
# !rm ../notebooks/apis/*.png

# %%
# !rm -r ../notebooks/apis/outdir
