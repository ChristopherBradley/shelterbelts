# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Download Meta Global Canopy Height tiles

# %%
import os
import shutil

import geopandas as gpd
import requests


# %%
# # NCI Filepaths
# canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height_v2'                                       # Initially empty directory
# canopy_height_aus_gpkg = '/g/data/xe2/datasets/Global_Canopy_Height/canopy_height_tiles_aus.gpkg'       # Created this by running demo_canopy_heihgt.py, and intersecting the resulting tiles_global.py with the aus boundary in QGIS.
# tiles_gpkg = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg'  # Created this with bounding_boxes.py


# %%
# Local filepaths
canopy_height_dir = '/home/christopher-bradley/repos/shelterbelts/tmpdir'
canopy_height_aus_gpkg = '/home/christopher-bradley/Documents/PHD/data/Outlines/canopy_height_tiles_v2_aus.gpkg'
tiles_gpkg = '/home/christopher-bradley/repos/shelterbelts/examples/classifications/tiff_footprints_years.gpkg'


# %%
# canopy_baseurl = 'https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/'  # v1
canopy_baseurl = 'https://s3.amazonaws.com/dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm/'  # v2

# %%
# Find the relevant tiles in Australia
gdf_canopy_height_aus = gpd.read_file(canopy_height_aus_gpkg)
gdf_target = gpd.read_file(tiles_gpkg).to_crs(gdf_canopy_height_aus.crs)

gdf_overlapping = (
    gpd.sjoin(
        gdf_canopy_height_aus,
        gdf_target[['geometry']],
        how='inner',
        predicate='intersects',
    )
    .drop_duplicates('tile')
)
tiles = list(gdf_overlapping['tile'])
print(f'Tiles overlapping target footprints: {len(tiles)}')

# %%
# Exclude any tiles we've already downloaded
to_download = [
    t for t in tiles
    if not os.path.isfile(os.path.join(canopy_height_dir, f'{t}.tif'))
]
print(f'Tiles still to download: {len(to_download)}')

# %%
# Just download a single tile for testing
to_download = to_download[:1]
to_download

# %%
for tile in to_download:
    url = f'{canopy_baseurl}{tile}.tif'
    filename = os.path.join(canopy_height_dir, f'{tile}.tif')
    if requests.head(url).status_code == 200:
        with requests.get(url, stream=True) as stream, open(filename, 'wb') as out:
            shutil.copyfileobj(stream.raw, out)
        print(f'Downloaded {filename}')

# %%
