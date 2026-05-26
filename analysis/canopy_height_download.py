# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
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
import time

import geopandas as gpd
import requests


# %%
# NCI Filepaths
canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height_v2'                                       # Initially empty directory
canopy_height_aus_gpkg = '/home/147/cb8590/Projects/shelterbelts/tiles_global.geojson'                  # Created this by running demo_canopy_height.py, and intersecting the resulting tiles_global.py with the aus boundary in QGIS. Note: need to make sure this matches the canopy_baseurl, since v1 used 60km tiles and v2 uses 30km tiles (will get a 404 error if it doesn't match).
tiles_gpkg = '/g/data/xe2/cb8590/Nick_outlines/barra_bboxs_grazing_no_bwh.gpkg'  # Grazing/no-BWH bboxes filtered from barra_trees_s4_aus_noxy_df_4326_2020


# # %%
# # Local filepaths
# canopy_height_dir = '/home/christopher-bradley/repos/shelterbelts/tmpdir'
# canopy_height_aus_gpkg = '/home/christopher-bradley/Documents/PHD/data/Outlines/canopy_height_tiles_v2_aus.gpkg'
# tiles_gpkg = '/home/christopher-bradley/repos/shelterbelts/examples/classifications/tiff_footprints_years.gpkg'


# %%
# canopy_baseurl = 'https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/'  # v1
canopy_baseurl = 'https://s3.amazonaws.com/dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm/'  # v2 (much better)

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
for tile in to_download:
    url = f'{canopy_baseurl}{tile}.tif'
    filename = os.path.join(canopy_height_dir, f'{tile}.tif')
    for attempt in range(5):
        try:
            if requests.head(url).status_code == 200:
                with requests.get(url, stream=True) as stream, open(filename, 'wb') as out:
                    shutil.copyfileobj(stream.raw, out)
                print(f'Downloaded {filename}')
            break
        except requests.exceptions.ConnectionError as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f'Connection error on {tile} (attempt {attempt + 1}), retrying in {wait}s: {e}')
                time.sleep(wait)
            else:
                print(f'Failed to download {tile} after 5 attempts: {e}')
    time.sleep(1)
