# +
# Download canopy height tiles matching each of Nick's LiDAR tiles
# -

import os
import geopandas as gpd
import requests
import shutil

canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'
outdir = '/scratch/xe2/cb8590/Nick_GCH'

# Load the Global_Canopy_Height footprints
canopy_height_aus_gpkg = '/g/data/xe2/datasets/Global_Canopy_Height/canopy_height_tiles_aus.gpkg'
gdf_canopy_height_aus = gpd.read_file(canopy_height_aus_gpkg)
len(gdf_canopy_height_aus)

# Load Nick's raster footprints
tiles_gpkg = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg'
gdf_nick_treecover = gpd.read_file(tiles_gpkg)
gdf_nick_treecover = gdf_nick_treecover.to_crs(gdf_canopy_height_aus.crs)
len(gdf_nick_treecover)



# Find just the tiles that overlap with one of Nick's tiles.
gdf_result = (
    gpd.sjoin(
        gdf_canopy_height_aus,
        gdf_nick_treecover[['geometry']],
        how='inner',
        predicate='intersects'
    )
    .drop_duplicates("tile")
)
tiles = list(gdf_result['tile'])
print("Length of tiles:", len(tiles))

# Create a list of tiles we haven't downloaded yet
to_download = []
for tile in tiles:
    tile_path = os.path.join(canopy_height_dir, f"{tile}.tif")
    if not os.path.isfile(tile_path):
        to_download.append(tile_path)
print("Length of to_download:", len(to_download))

# +
to_download = []
for tile in tiles:
    tile_path = os.path.join(canopy_height_dir, f"{tile}.tif")
    if not os.path.isfile(tile_path):
        to_download.append(tile)
if len(to_download) == 0:
    print("Nothing to download")

canopy_baseurl = "https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/"

# -

# to_download = to_download[:2]

# +
# tile = to_download[0]
# url = canopy_baseurl + f'{tile}.tif'
# filename = os.path.join(canopy_height_dir, f'{tile}.tif')
# response = requests.head(url)
# if response.status_code == 200:
#     with requests.get(url, stream=True) as stream:
#         with open(filename, "wb") as file:
#             shutil.copyfileobj(stream.raw, file)
#     print(f"Downloaded {filename}")
# -

print("About to_download tiles:", len(to_download))
for tile in to_download:
    url = canopy_baseurl + f'{tile}.tif'
    filename = os.path.join(canopy_height_dir, f'{tile}.tif')
    response = requests.head(url)
    if response.status_code == 200:
        with requests.get(url, stream=True) as stream:
            with open(filename, "wb") as file:
                shutil.copyfileobj(stream.raw, file)
        print(f"Downloaded {filename}")
