# This is a nice example of running things in parallel, 
# However, for the canopy height I ended up just running canopy_height_aus.pbs because it only took 9 hours.

import os
import glob
import pandas as pd
from pyproj import Transformer
import rioxarray as rxr
import ast
import traceback, sys
import time
from rasterio.enums import Resampling

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

from shelterbelts.apis.canopy_height import canopy_height_bbox

tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"
tmp_dir = '/scratch/xe2/cb8590/tmp'
canopy_height_dir ='/scratch/xe2/cb8590/Global_Canopy_Height'
outdir = '/scratch/xe2/cb8590/Nick_GCH'

# Basic setup so I can run this script from the command with different arguments
# This allows me to parallelise by submitting multiple jobs from a bash script since trying to parallelise with concurrent.futures didn't do anything, probably due to API rate limiting
import argparse
import logging
logging.basicConfig(level=logging.INFO)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Run the canopy height download for a bunch of pre-calculated locations
        
Example usage locally:
python3 canopy_height_batch.py --csv batch.csv""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--csv", type=str, required=True, help="csv filename containing rows of bboxs to download")
    return parser.parse_args()


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


# I ran this code in a jupyter notebook on NCI ARE before submitting the jobs via qsub
# I want to generalise this and make it work on gpkg files instead of csv's for the final pipeline
def generate_batches():
    csv = os.path.join(outlines_dir, "nick_bbox_crs.csv")
    df = pd.read_csv(csv)

    # Randomising so I get a nice distribution of tiles even before all of them have been download
    df_randomised = df.sample(len(df),random_state=0)

    # Technically I can launch 1000 jobs at once but choosing 100 to hammer the scheduler a little less, and it should only take 14 hours this way anyway.
    num_batches = 100
    batch_size = len(df_randomised) // num_batches + 1
    batches = [df_randomised.iloc[i:i + batch_size] for i in range(0, len(df_randomised), batch_size)]

    for i, df_batch in enumerate(batches):
        filename = os.path.join(outlines_dir, "100_batches", f"batch_{i}.csv")
        df_batch.to_csv(filename, index=False)
        print("Saved", filename)
        
    df_batches = pd.DataFrame(glob.glob(os.path.join(outlines_dir,  "100_batches", '*')), columns=["batch"])
    filename = '/g/data/xe2/cb8590/Nick_outlines/canopy_height_batches.csv'
    df_batches.to_csv(filename, index=False, header=False)
    print("Saved", filename)
    
    df_batches[:1].to_csv('/g/data/xe2/cb8590/Nick_outlines/canopy_height_batchesx1.csv', index=False, header=False)
    df_batches[:2].to_csv('/g/data/xe2/cb8590/Nick_outlines/canopy_height_batchesx2.csv', index=False, header=False)
    df_batches[:4].to_csv('/g/data/xe2/cb8590/Nick_outlines/canopy_height_batchesx4.csv', index=False, header=False)


# Example args for debugging
# args = argparse.Namespace(
#     csv='/g/data/xe2/cb8590/Nick_outlines/100_batches/batch_0.csv'
# )
args = parse_arguments()


# Load the tile bounding boxes
df = pd.read_csv(args.csv)

# Remove any tiffs we've already run
already_downloaded = glob.glob(f'/scratch/xe2/cb8590/Nick_GCH/*')
print("Number of tiles already downloaded:", len(already_downloaded))

tile_ids = ["_".join(tile.split('/')[-1].split('_')[:2]) for tile in already_downloaded]
df_new = df[~df['Tile'].isin(tile_ids)]
print("Number of tiles to download:", len(df_new))

rows = df_new.values.tolist()

# %%time
# Just running sequentially because I didn't see any speed up when running in parallel
for row in rows:
    try:
        global_canopy_height_tile(row)
    except Exception as e:
        print(f"Failed {row[0]}")
# +
# Benchmarking:       
# Took 54 mins for 10 downloads that in theory happened in parallel
# Took 33 mins for the next 10 downloads that in theory happened in parallel. 
# Took 2 hours for next 20 downloads that in theory happened in parallel.

# Doesn't seem like it's happening in parallel. So, assuming 10 per hour, it would take 1400 hours = way too long to do all of them. 
# I think my next best bet is to make a bash script to submit something like 100 separate jobs. That would take about 14 hours which is much more reasonable. 
# I want to try 2 jobs in parallel for 14 hours before doing 100 jobs in parallel for 14 hours.

# Filesize is about 20GB for 50 tiles. Estimating about 10k tiles which would be about 4TB. Yikes quite a lot, but also well within scratch limits so that's fine.

