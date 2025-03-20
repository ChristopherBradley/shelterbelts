# +
# Create 5km woody veg tiles to match the sentinel training data
# -

import os
import glob
import pickle
import rioxarray as rxr

outdir = "/g/data/xe2/cb8590/shelterbelts/"

# Load the woody veg tiff file
filename = os.path.join(outdir, "Tas_WoodyVeg_201903_v2.2.tif")
ds = rxr.open_rasterio(filename)

tiles = glob.glob("/g/data/xe2/cb8590/shelterbelts/*_ds2.pkl")

# +
# %%time
# For each tile, load the satellite imagery and use reproject_match to clip the woodyveg tif
# (Would be more computationally efficient to not load the satellite imagery, but the code is simpler this way)

for tile in tiles:    
    # Load the satellite imagery for this tile
    with open(tile, 'rb') as file:
        ds2 = pickle.load(file)
        
    # reproject_match
    ds_matched = ds.rio.reproject_match(ds2)
    
    # Save the cropped tif
    stub = tile.replace(outdir,"").replace("_ds2.pkl","")
    filename = os.path.join(outdir, f"{stub}_woodyveg_2019.tif")
    ds_matched.rio.to_raster(filename)
    print("Saved", filename)
# -


