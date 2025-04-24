# +
import glob
import rasterio
import pandas as pd
import ast
import traceback, sys

import os
import rioxarray as rxr

from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed
# -

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

from shelterbelt_identification.worldcover import worldcover_bbox


def extract_bbox_crs():
    """Extract the bbox and crs for each of Nick's tiles"""
    rows = []
    for filename in tree_cover_tiles:
        tile_id = "_".join(filename.split('/')[-1].split('_')[:2])
        with rasterio.open(filename) as src:
            bounds = src.bounds
            crs = src.crs.to_string()
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        rows.append([tile_id, bbox, crs])

    # Save this as a csv file for later
    df = pd.DataFrame(rows, columns=["Tile", "bbox", "crs"])
    filename = os.path.join(outlines_dir, "nick_bbox_crs.csv")
    df.to_csv(filename, index=False)
    print("Saved", filename)
    # Took 5 mins


def tree_cover_tile(row):
    # Save a tif file matching the bbox and crs for each of Nick's tree cover tiff files
    stub, bbox, crs = row
    try:
        filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{stub}_binary_tree_cover_10m.tiff'
        ds_tree_cover = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

        bbox = ast.literal_eval(bbox) # The bbox is saved as a string initially because I stored it in a csv file with pandas

        ds_worldcover, bbox_4326 = worldcover_bbox(bbox, crs, worldcover_dir, stub)
        ds_worldcover_28355 = ds_worldcover.rio.reproject_match(ds_tree_cover)

        filename = f'/scratch/xe2/cb8590/Nick_worldcover_reprojected/{stub}_worldcover.tif'
        ds_worldcover_28355.rio.to_raster(filename)
        print("Saved", filename)

        filename = f'/scratch/xe2/cb8590/Nick_worldcover_trees/{stub}_worldcover_trees.tif'
        ds_worldcover_trees = (ds_worldcover_28355 == 10).astype(int)
        ds_worldcover_28355.rio.to_raster(filename)
        print("Saved", filename)
        
    except Exception as e:
        print(f"Error in worker {stub}:", flush=True)
        traceback.print_exc(file=sys.stdout)
        raise


# +
tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
worldcover_dir = "/scratch/xe2/cb8590/Nick_worldcover_4326"
outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"

tree_cover_tiles = glob.glob(f'{tree_cover_dir}/*.tiff')
# -

# Load the bounding boxes for each tiff file
filename = os.path.join(outlines_dir, "nick_bbox_crs.csv")
df = pd.read_csv(filename)
rows = df.values.tolist()
batches = [rows[i:i + 100] for i in range(0, len(rows), 100)]

# +
# # Redo the failed downloads. Would have been better to have saved these in a list instead of having to read the error logs afterwards
# failed_worldcover_downloads = [
# "g2_51930",
# "g1_03949",
# "g2_53970",
# "g2_52577",
# "g2_61925",
# "g1_03891",
# "g1_04348",
# "g2_2611",
# "g2_61683",
# "g1_05231"
# ]
# df = df[df["Tile"].isin(failed_worldcover_downloads)]
# rows = df.values.tolist()
# batches = [rows[i:i + 1] for i in range(0, len(rows), 1)]
# -

# Extra line for running a single tile so I don't accidentally run them all again
test = [
"g2_51930"
]
df = df[df["Tile"].isin(test)]
rows = df.values.tolist()
batches = [rows[i:i + 1] for i in range(0, len(rows), 1)]

batches

# +
# %%time

# Just use 100 workers at a time or else the planetary computer gets overwhelmed. 
for i, batch in enumerate(batches):
    rows = batch
    with ProcessPoolExecutor(max_workers=len(rows)) as executor:
        print(f"Starting {len(rows)} workers for batch {i}")
        futures = [executor.submit(tree_cover_tile, row) for row in rows]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with: {e}", flush=True)

# Took 10 mins to download 1000 tiles
# Should take about 2 hours 20 mins to download the full 14k tiles
# -


