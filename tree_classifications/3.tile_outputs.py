# +
# Create 5km woody veg and canopy cover tiles to match the sentinel training data
# -

import os
import glob
import pickle
import rioxarray as rxr

outdir = "/g/data/xe2/cb8590/shelterbelts/"

# +
# Load the woody veg or canopy cover tiff file
# filename = os.path.join(outdir, "Tas_WoodyVeg_201903_v2.2.tif")  # Binary classifications
# sub_stub = "woodyveg"

filename = os.path.join(outdir, "Tas_CanopyCover_201903_v2.2.tif")  # Continuous classifications
sub_stub = "canopycover"

ds = rxr.open_rasterio(filename)
# -

tiles = glob.glob("/g/data/xe2/cb8590/shelterbelts/*_ds2.pkl")

# +
# %%time
# For each tile, load the satellite imagery and use reproject_match to clip the woodyveg tif
# (Would be more computationally efficient to not load the satellite imagery, but the code is much simpler this way)

for tile in tiles:    
    # Load the satellite imagery for this tile
    with open(tile, 'rb') as file:
        ds2 = pickle.load(file)
        
    # reproject_match
    ds_matched = ds.rio.reproject_match(ds2)
    
    # Save the cropped tif
    stub = tile.replace(outdir,"").replace("_ds2.pkl","")
    filename = os.path.join(outdir, f"{stub}_{sub_stub}_2019.tif")
    ds_matched.rio.to_raster(filename)
    print("Saved", filename)

# Took 5 mins
# -


