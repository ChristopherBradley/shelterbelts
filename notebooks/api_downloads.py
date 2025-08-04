# +
# Demo-ing each of the data downloads
# Note: I've predownloaded all of these datasets to NCI (except roads atm), so I shouldn't need to wait for these API calls when scaling up

# +
# Worldcover download
from shelterbelts.apis.worldcover import worldcover, visualise_categories, worldcover_cmap, worldcover_labels

outdir = '../outdir'
stub = 'api_downloads'

lat = -34.37825
lon = 148.42490
buffer = 0.012   # In degrees in a single direction. For example, 0.01 degrees is about 1km so it would give a 2kmx2km area.
# -

# %%time
ds_worldcover = worldcover(lat, lon, buffer, outdir, stub)

visualise_categories(ds_worldcover['worldcover'], None, worldcover_cmap, worldcover_labels, "ESA WorldCover")

# Tolan canopy height download
from shelterbelts.apis.canopy_height import canopy_height, visualise_canopy_height
tmpdir = '../tmpdir'

# %%time
ds_tolan = canopy_height(lat, lon, buffer, outdir, stub, tmpdir)

visualise_canopy_height(ds_tolan)

# Converting to tree vs no tree
from shelterbelts.util.binary_trees import worldcover_trees, canopy_height_trees, cmap_woody_veg, labels_woody_veg
da_worldcover_trees = worldcover_trees('../outdir/api_downloads_worldcover.tif', outdir)
da_canopy_height_trees = canopy_height_trees('../outdir/api_downloads_canopy_height.tif', outdir)

visualise_categories(da_worldcover_trees, None, cmap_woody_veg, labels_woody_veg, "ESA WorldCover Trees")

visualise_categories(da_canopy_height_trees, None, cmap_woody_veg, labels_woody_veg, "Tolan Canopy Height Trees")

# BARRA wind download
from shelterbelts.apis.barra_daily import barra_daily, wind_rose
variables = ["uas", "vas"]
start_year = 2020 
end_year = 2021

# %%time
ds_wind = barra_daily(variables, lat, lon, buffer, start_year, end_year, outdir, stub)

wind_rose(ds_wind)

# Open Streetmap Roads
from shelterbelts.apis.osm import osm_roads, roads_cmap, roads_labels

# %%time
gdf, ds = osm_roads('../outdir/api_downloads_worldcover.tif', outdir, stub)

visualise_categories(ds['roads'], None, roads_cmap, roads_labels, "Open Street Map Roads")

# +
# from shelterbelts.apis. import 
# Catchments
# -

# Hydrolines
