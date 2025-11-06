# +
import os
import shutil
import re
import math
import glob
from pathlib import Path

import xarray as xr
import numpy as np

import geopandas as gpd
from shapely.prepared import prep
from shapely.geometry import box, Polygon, MultiPolygon


# -

def mosaic_subfolders(base_str='/scratch/xe2/cb8590/barra_trees_s4_2024'):
    """Create subfolders for mosaicking tiles"""
    output_str = f"{base_str}/subfolders"
    base_dir = Path(base_str)
    output_base = Path(output_str)
    
    # Create the output base if needed
    output_base.mkdir(exist_ok=True)

    # Approx tile size (degrees): 4 km ≈ 0.036°, so 50 tiles ≈ 1.8° — round to 2°
    block_size_deg = 2.0

    # pattern = re.compile(r"(\d+_\d+)-(\d+_\d+)_predicted\.tif") # regex for "28_21-153_54_predicted.tif"
    pattern = re.compile(r"(\d+_\d+)-(\d+_\d+)_y\d+_predicted\.tif")  # regex for 28_21-153_54_y2024_predicted.tif

    for i, tif_path in enumerate(base_dir.glob("*.tif")):
        if i % 1000 == 0:
            print(f"Moving tile {i}")
        m = pattern.match(tif_path.name)
        if not m:
            continue

        lat_str, lon_str = m.groups()
        lat = float(lat_str.replace("_", "."))
        lon = float(lon_str.replace("_", "."))

        # Find the 2° bin indices
        lat_bin = math.floor(lat / block_size_deg) * block_size_deg
        lon_bin = math.floor(lon / block_size_deg) * block_size_deg

        # Folder name e.g. lat_-28_lon_152
        subfolder = output_base / f"lat_{lat_bin:.0f}_lon_{lon_bin:.0f}"
        subfolder.mkdir(exist_ok=True)

        # Move file (or use shutil.copy2 if you prefer copying)
        shutil.move(str(tif_path), subfolder / tif_path.name)

    # Took 30 secs for 50k tiles

# +
# # %%time
# year = 2017
# years = [2019,2020, 2021, 2022, 2023, 2024]
# for year in years:
#     mosaic_subfolders(f'/scratch/xe2/cb8590/barra_trees_s4_{year}_actnsw_4326')

# +
# I run this function in a notebook to prep the sh file that does the qsubs in parallel
non_suffixes=['_confidence50', '_confidence50_fixedlabels', '_corebugfix']
non_contains = ['linear_tifs', 'merged_predicted']
folder_with_subfolders = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders'
def get_subfolders(folder_with_subfolders, non_suffixes=[], non_contains=[]):
    """Find all the original subfoldes in the larger folder"""
    folders = glob.glob(f'{folder_with_subfolders}/*')
    
    # I should have better folder management so I don't need to jump through all these hurdles
    folders = [f for f in folders if 
             os.path.isdir(f)
             and not any(f.endswith(non_suffix) for non_suffix in non_suffixes)
             and not any(non_contain in f for non_contain in non_contains)
            ]
    stems = [Path(folder).stem for folder in folders]
    stems_string = " ".join(stems)
    return stems_string

# get_subfolders('/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/subfolders/')


# +
import argparse
def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('folder', type=str, help='Folder containing lots of tifs that we want to move into subfolders')

    return parser.parse_args()


# -

if __name__ == '__main__':
    args = parse_arguments()
    folder = args.folder
    print(f"Mosaicking folder: {folder}")
    mosaic_subfolders(folder)
    stems_string = get_subfolders(os.path.join(folder, "subfolders"))
    print(f"Created subfolders:\n{stems_string}")

# +
# folder='/scratch/xe2/cb8590/barra_trees_s4_2020_actnsw_4326_weightings'
# stems_string = get_subfolders(os.path.join(folder, "subfolders"))
# print(f"Created subfolders:\n{stems_string}")
