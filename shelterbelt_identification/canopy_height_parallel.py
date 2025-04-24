import os
import pandas as pd
from pyproj import Transformer
import rioxarray as rxr
import ast
import traceback, sys
import time
from rasterio.enums import Resampling
from concurrent.futures import ProcessPoolExecutor, as_completed

# Change directory to this repo
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:
    repo_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from repo root
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)
print(f"Running from {repo_dir}")

from shelterbelt_identification.canopy_height import canopy_height_bbox


def global_canopy_height_tile(row):
    # Save a tif file matching the bbox and crs for each of Nick's tree cover tiff files
    stub, bbox, crs = row
    try:
        filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{stub}_binary_tree_cover_10m.tiff'
        ds_tree_cover = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

        bbox = ast.literal_eval(bbox) # The bbox is saved as a string initially because I stored it in a csv file with pandas
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        minx, miny = transformer.transform(bbox[0], bbox[1])
        maxx, maxy = transformer.transform(bbox[2], bbox[3])
        bbox_4326 = [minx, miny, maxx, maxy] # Need to convert to epsg_4326 to match how it's stored in the global canopy height tiles

        filename = canopy_height_bbox(bbox_4326, outdir, stub, tmp_dir, canopy_height_dir)
        
        # Load the tiff we just downloaded, and match the resolution of Nick's tiffs
        time.sleep(1)
        ds_gch = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
        ds_gch_10m = ds_gch.rio.reproject_match(ds_tree_cover, Resampling.max)
            
        filename = os.path.join(outdir, f'{stub}_10m.tif')
        ds_gch_10m.rio.to_raster(filename)
        # print("Saved", filename)
        
    except Exception as e:
        print(f"Error in worker {stub}:", flush=True)
        traceback.print_exc(file=sys.stdout)
        raise


tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"
tmp_dir = '/scratch/xe2/cb8590/tmp'
canopy_height_dir ='/scratch/xe2/cb8590/Global_Canopy_Height'
outdir = '/scratch/xe2/cb8590/Nick_GCH'

# Load the tile bounding boxes
filename = os.path.join(outlines_dir, "nick_bbox_crs.csv")
df = pd.read_csv(filename)
rows = df.values.tolist()
num_cpus_per_batch = 20
batches = [rows[i:i + num_cpus_per_batch] for i in range(0, len(rows), num_cpus_per_batch)]
batches = batches[1:2]

# +
# %%time
# Run all the tiles in batches
for i, batch in enumerate(batches):
    rows = batch
    with ProcessPoolExecutor(max_workers=len(rows)) as executor:
        print(f"Starting {len(rows)} workers for batch {i+1}")
        futures = [executor.submit(global_canopy_height_tile, row) for row in rows]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with: {e}", flush=True)
                
# Took 54 mins for 10 downloads that in theory happened in parallel
# Took 33 mins for the next 10 downloads that in theory happened in parallel
# -


