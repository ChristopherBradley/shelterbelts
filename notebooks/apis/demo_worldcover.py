# %% [markdown]
# # WorldCover API Demo

# %%
from shelterbelts.apis.worldcover import worldcover, worldcover_cmap, worldcover_labels

# %% [markdown]
# ## Default Parameters
# The default location is a ~2km x 2km region near Milgadara, NSW, Australia. It should take about 10 secs to load.

# %%
# %%time
ds = worldcover()
ds

# %% [markdown]
# ## Visualisating the results
# You can view the results inside the notebook, or in preview as a PNG or TIF, or properly geolocated in QGIS.

# %%
from shelterbelts.utils.visualisation import visualise_categories

visualise_categories(ds['worldcover'], colormap=worldcover_cmap, labels=worldcover_labels, title="Default Values")

# %% [markdown]
# ## Changing Buffer Size
# You can adjust the buffer parameter to change the size of the downloaded region. The buffer is in degrees (~1 degree ≈ 100 km depending on the latitude) in each direction from the centre point. A 10km x 10km region should take about 1 min to load.

# %%
# %%time
ds_10km = worldcover(buffer=0.05)
visualise_categories(ds_10km['worldcover'], colormap=worldcover_cmap, labels=worldcover_labels, title="Larger Buffer")

# %% [markdown]
# ## Changing Location
# Use the lat and lon arguments to download data for a different location. Here is an example of ANU in Canberra.

# %%
ds_canberra = worldcover(lat=-35.287, lon=149.117)
visualise_categories(ds_canberra['worldcover'], colormap=worldcover_cmap, labels=worldcover_labels, title="Different Location")

# %% [markdown]
# ## Changing Output Directory
# Specify a custom output directory for the saved files.

# %%
import os
outdir = "outdir"
os.makedirs(outdir, exist_ok=True)
ds = worldcover(outdir=outdir)

# %% [markdown]
# ## Changing Filename Prefix
# Choose a stub to be used as the prefix for output filenames.

# %%
ds = worldcover(stub="DEMO", outdir=outdir)

# %% [markdown]
# ## Disabling GeoTIFF Output
# Set `save_tif=False` to skip saving the GeoTIFF file and only keep the dataset in memory. Useful for when running within a larger pipeline.

# %%
ds = worldcover(save_tif=False)

# %% [markdown]
# ## Disabling Visualisation Plot
# Set `plot=False` to skip generating and saving the PNG visualisation. This can be used in combination with `savetif=False` to avoid any outputs.

# %%
ds = worldcover(plot=False)

# %% [markdown]
# ## Command Line Interface
# You can also use the function from the command line with the same defaults and parameters.

# %%
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

# %%
# !python shelterbelts/apis/worldcover.py --help

# %%
# !python shelterbelts/apis/worldcover.py

# %%
# !python shelterbelts/apis/worldcover.py --lat -35.287 --lon 149.117 --buffer 0.02 --stub command_line --outdir ../notebooks/apis/outdir

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# !rm ../notebooks/apis/*.tif
# !rm ../notebooks/apis/*.png

# %%
# !rm -r ../notebooks/apis/outdir
