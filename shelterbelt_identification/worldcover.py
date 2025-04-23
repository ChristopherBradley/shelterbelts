# +
# Example code is here: https://planetarycomputer.microsoft.com/dataset/esa-worldcover#Example-Notebook

# +
# %%time
import os
import odc.stac
import pystac_client
import planetary_computer
import rioxarray as rxr

import matplotlib.colors
from matplotlib import cm
import matplotlib.pyplot as plt

from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed

# -

world_cover_layers = {
    "Tree cover": 10, # Green
    "Shrubland": 20, # Orange
    "Grassland": 30, # Yellow
    "Cropland": 40, # pink
    "Built-up": 50, # red
    "Permanent water bodies": 80, # blue
}


def worldcover_bbox(bbox=[147.735717, -42.912122, 147.785717, -42.862122], crs="EPSG:4326", outdir=".", stub="Test"):
    """Download worldcover data for the region of interest"""
    
    # Need to have the bbox in EPSG:4326 for the catalog search
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    bbox_4326 = [minx, miny, maxx, maxy]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox_4326
    )
    items = list(search.items())
    items = [items[0]]
    ds = odc.stac.load(items, crs="EPSG:4326", bbox=bbox_4326)
    ds_map = ds.isel(time=0)['map']

    filename = os.path.join(outdir, f"{stub}_worldcover.tif")
    ds_map.rio.to_raster(filename)
    print("Downloaded", filename)

    return ds_map, bbox_4326


def worldcover_centerpoint(lat=-34.3890427, lon=148.469499, buffer=0.05, outdir=".", stub="Test"):
    """Download worldcover data for the region of interest"""
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    crs="EPSG:4326"
    ds_map = worldcover_bbox(bbox, crs, outdir, stub)
    return ds_map, bbox


def visualise_worldcover(ds, bbox, outdir=".", stub="Test"):
    """Pretty visualisation using the worldcover colour scheme"""
    
    # Download the colour scheme
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox,
    )
    items = list(search.items())
    items = [items[0]]
    
    class_list = items[0].assets["map"].extra_fields["classification:classes"]
    classmap = {
        c["value"]: {"description": c["description"], "hex": c["color-hint"]}
        for c in class_list
    }

    # Prep the colour bar
    colors = ["#000000" for r in range(256)]
    for key, value in classmap.items():
        colors[int(key)] = f"#{value['hex']}"
    cmap = matplotlib.colors.ListedColormap(colors)
    
    values = [key for key in classmap]
    boundaries = [(values[i + 1] + values[i]) / 2 for i in range(len(values) - 1)]
    boundaries = [0] + boundaries + [255]
    ticks = [(boundaries[i + 1] + boundaries[i]) / 2 for i in range(len(boundaries) - 1)]
    tick_labels = [value["description"] for value in classmap.values()]
    
    normalizer = matplotlib.colors.Normalize(vmin=0, vmax=255)

    # Plot the Map
    fig, ax = plt.subplots(figsize=(16, 14))
    ds.plot(
        ax=ax, cmap=cmap, norm=normalizer
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=normalizer, cmap=cmap),
        boundaries=boundaries,
        values=values,
        cax=fig.axes[1].axes,
    )
    colorbar.set_ticks(ticks, labels=tick_labels)

    filename = os.path.join(outdir, stub)
    plt.savefig(filename)
    print("Saved", filename)


# +
# # %%time
# if __name__ == '__main__':
    
#     # Change directory to this repo
#     import os, sys
#     repo_name = "shelterbelts"
#     if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
#         repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
#     elif os.path.basename(os.getcwd()) != repo_name:
#         repo_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
#     else:  # Already running locally from repo root
#         repo_dir = os.getcwd()
#     os.chdir(repo_dir)
#     sys.path.append(repo_dir)
#     print(f"Running from {repo_dir}")

#     # Coords for Fulham: -42.887122, 147.760717
#     lat=-42.887122
#     lon=147.760717
#     buffer=0.025
#     bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
#     ds, bbox = worldcover_centerpoint(lat=lat, lon=lon, buffer=buffer, outdir="data", stub="Fulham")
#     visualise_worldcover(ds, bbox)
# -

import glob
import rasterio
import pandas as pd
import ast
import traceback, sys

# +
tree_cover_dir = "/g/data/xe2/cb8590/Nick_Aus_treecover_10m"
worldcover_dir = "/scratch/xe2/cb8590/Nick_worldcover"
outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"

tree_cover_tiles = glob.glob(f'{tree_cover_dir}/*.tiff')


# +
# %%time
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
    
# df = extract_bbox_crs()


# +
# %%time
def tree_cover_tile(row):
    # Save a tif file matching the bbox and crs for each of Nick's tree cover tiff files
    stub, bbox, crs = row
    try:
        filename = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{stub}_binary_tree_cover_10m.tiff'
        ds_tree_cover = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

        bbox = ast.literal_eval(bbox) # The bbox is saved as a string initially because I stored it in a csv file with pandas

        ds_worldcover, bbox_4326 = worldcover_bbox(bbox, crs, worldcover_dir, stub)
        ds_worldcover_28355 = ds_worldcover.rio.reproject_match(ds_tree_cover)

        filename = f'/scratch/xe2/cb8590/{stub}_worldcover.tif'
        ds_worldcover_28355.rio.to_raster(filename)
        # print("Saved", filename)

        filename = f'/scratch/xe2/cb8590/{stub}_worldcover_trees.tif'
        ds_worldcover_trees = (ds_worldcover_28355 == 10).astype(int)
        # print("Saved", filename)
        
    except Exception as e:
        print(f"Error in worker {stub}:", flush=True)
        traceback.print_exc(file=sys.stdout)
        raise

# tree_cover_tile(row)
# -

### I should move these worldcover tifs out of my main scratch and into a separate folder within scratch. And I should start using the tmp folder within scratch more consistently


# Load the bounding boxes for each tiff file
filename = os.path.join(outlines_dir, "nick_bbox_crs.csv")
df = pd.read_csv(filename)
rows = df.values.tolist()
batches = [rows[i:i + 100] for i in range(0, len(rows), 100)]

batches = batches[10:]

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


