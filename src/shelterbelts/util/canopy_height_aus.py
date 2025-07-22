# +
# Downloading canopy height for all of Australia
# I downloaded all of Worldcover manually from https://viewer.esa-worldcover.org/worldcover (can only do 100 tiles at a time, so it took 2 downloads)

import geopandas as gpd
import argparse
# -

# Change directory to this repo - this should work on gadi or locally via python or jupyter.
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
src_dir = os.path.join(repo_dir, 'src')
os.chdir(src_dir)
sys.path.append(src_dir)
print(src_dir)

from shelterbelts.apis.canopy_height import download_new_tiles


def filter_canopy_height():
    """Used this code to generate the canopy_height_tiles_aus.gpkg"""
    # Downloaded from here: https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/tiles.geojson
    canopy_height_tiles = '/Users/christopherbradley/repos/PHD/shelterbelts/tiles_global.geojson' 
    
    # Downloaded from here: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
    aus_boundary = '/Users/christopherbradley/Documents/PHD/Data/AUS_2021_AUST_SHP_GDA2020/AUS_2021_AUST_GDA2020.shp'
    
    outpath = '/Users/christopherbradley/Documents/PHD/Data/Global_Canopy_Height/canopy_height_tiles_aus.gpkg'

    # Load the canopy height and australia boundary
    df_canopy_height = gpd.read_file(canopy_height_tiles)
    df_aus_boundary = gpd.read_file(aus_boundary)
    df_aus_boundary = df_aus_boundary.to_crs(df_canopy_height.crs)

    # Filter the canopy height tiles and save a geopackage
    aus_geom = df_aus_boundary.iloc[0].geometry
    df_canopy_height_filtered = df_canopy_height[df_canopy_height.intersects(aus_geom)]
    df_canopy_height_filtered.to_file(outpath)

    # Then copied this canopy_height_tiles_aus.gpkg to gadi with scp

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    # Load the list of all tiles in Australia
    canopy_height_tiles = '/g/data/xe2/cb8590/Nick_outlines/canopy_height_tiles_aus.gpkg'
    df_canopy_height = gpd.read_file(canopy_height_tiles)
    tiles = list(df_canopy_height['tile'])
    print("Tiles in Australia: ", len(tiles))

    # +
    # Create a list of tiles we haven't downloaded yet
    canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height/'

    to_download = []
    for tile in tiles:
        tile_path = os.path.join(canopy_height_dir, f"{tile}.tif")
        if not os.path.isfile(tile_path):
            to_download.append(tile)
    print("Tiles to download: ", len(to_download))
    # -

    args = parse_arguments()
    if args.limit:
        limit = int(args.limit)
        to_download = to_download[:limit]
        print("Limiting tiles to just: ", len(to_download))

    download_new_tiles(to_download, canopy_height_dir)
