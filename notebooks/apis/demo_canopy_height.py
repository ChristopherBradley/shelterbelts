# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Canopy Height API Demo

# %%
from shelterbelts.apis.canopy_height import canopy_height

# %% [markdown]
# ## Default Parameters
# The default location is a roughly 1km x 1km region near Milgadara, NSW, Australia. 
# The first download can take a few minutes as tiles need to be fetched from AWS (~200MB per tile).

# %%
# %%time
ds = canopy_height()

# %% [markdown]
# ## Visualising the Results
# You can view the results inside the notebook, or in preview as a PNG or TIF, or properly geolocated in QGIS.

# %%
from shelterbelts.apis.canopy_height import visualise_canopy_height

visualise_canopy_height(ds)

# %% [markdown]
# ## Changing Buffer Size
# You can adjust the buffer parameter to change the size of the downloaded region. 
# The buffer is in degrees (~1 degree ≈ 100 km depending on the latitude) in each direction from the centre point.

# %%
ds_larger = canopy_height(buffer=0.02)
visualise_canopy_height(ds_larger)

# %% [markdown]
# ## Changing Location
# Use the lat and lon arguments to download data for a different location.

# %%
ds_laverstock = canopy_height(lat=-34.68, lon=148.96)
visualise_canopy_height(ds_laverstock)

# %% [markdown]
# ## Changing Output Directory
# Specify a custom output directory for the saved files.

# %%
import os
outdir = "outdir"
os.makedirs(outdir, exist_ok=True)
ds = canopy_height(outdir=outdir)

# %% [markdown]
# ## Changing Filename Prefix
# Choose a stub to be used as the prefix for output filenames.

# %%
ds = canopy_height(stub="DEMO")

# %% [markdown]
# ## Changing Temporary Directory
# You can specify a custom temporary directory to cache these tiles downloaded from AWS for reuse. Otherwise they just get cached in the working directory.

# %%
tmpdir = "tmpdir"
os.makedirs(tmpdir, exist_ok=True)
ds = canopy_height(tmpdir=tmpdir)

# %% [markdown]
# ## Disabling GeoTIFF Output
# Set `save_tif=False` to skip saving the GeoTIFF file. Useful for when running within a larger pipeline.

# %%
ds = canopy_height(save_tif=False)

# %% [markdown]
# ## Disabling Visualisation Plot
# Set `plot=False` to skip generating and saving the PNG visualisation. 
# This can be used in combination with `save_tif=False` to avoid any outputs.

# %%
ds = canopy_height(plot=False)

# %% [markdown]
# ## Command Line Interface
# You can also use the function from the command line with the same defaults and parameters.

# %%
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

# %%
# !python shelterbelts/apis/canopy_height.py --help

# %%
# !python shelterbelts/apis/canopy_height.py --lat -35.287 --lon 149.117 --buffer 0.01 --stub command_line --outdir ../notebooks/apis/outdir --tmpdir ../notebooks/apis/tmpdir

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# !rm ../notebooks/apis/*.tif
# !rm ../notebooks/apis/*.png
# !rm ../notebooks/apis/*.geojson

# %%
# !rm -r ../notebooks/apis/outdir

# %%
# !rm -r ../notebooks/apis/tmpdir
