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
# # Attaching the acquisition year to tile bounding boxes
#
# Creates ``tiff_footprints_years.gpkg`` with one row per tiff tile showing the most
# recent LiDAR acquisition year and its bounding box in EPSG:3857.
#
# I started by running shelterbelts.classifications.bounding_boxes.py on the full folder of 14k tiles provided by Nick Pucino, and then intersected that with 'lidar_datasets_raw_4326_NoLandgate.gpkg' (also provided) to create 'lidar_clipped.gpkg' which is the input to this script. I have not committed any of these geopackages to the repo as the smallest one lidar_clipped.gpkg is 75MB.

# %%
import re

import geopandas as gpd
from shapely.ops import unary_union

# %% [markdown]
# ## 1. Read the clipped LiDAR footprints

# %%
filepath_lidar_gpkg = 'lidar_clipped.gpkg'  # Local
# filepath_lidar_gpkg = '/g/data/xe2/cb8590/Nick_outlines/lidar_clipped.gpkg'  # NCI

# %%
gdf = gpd.read_file(filepath_lidar_gpkg)
print(f'{len(gdf)} rows, {gdf["filename"].nunique()} unique tiff tiles')

# %% [markdown]
# ## 2. Consolidate the name columns
#
# The ELVIS metadata spreads the dataset name across four columns depending
# on file type (AHD, ORT, LAS, generic).

# %%
name_cols = ['object_name_ahd', 'object_name_ort', 'object_name_las', 'object_name']
gdf['name'] = gdf[name_cols].bfill(axis=1).iloc[:, 0]
print(f'{gdf["name"].isna().sum()} rows with no name (and hence no obvious year)')

# %% [markdown]
# ## 3. Extract the acquisition year
#
# The year is always the first ``20XX`` run of digits in the dataset name.
# Two datasets use non-standard prefixes and need special-casing.

# %%
def extract_year(name):
    if not isinstance(name, str):
        return None
    if name.startswith('Laura22021'):
        return 2021
    if name.startswith('Herbert1Lidar2020') or name.startswith('Herbert2Lidar2020'):
        return 2020
    m = re.search(r'20\d\d', name)
    return int(m.group()) if m else None

examples = [
    ('ACT2015_4ppm-C3-AHD_6626058_55_0002_0002.zip',               2015),
    ('Yarrangobilly201803-LID2-C3-AHD_6106040_55_0002_0002.zip',    2018),
    ('BelyandoCrossing_2013_Loc_SW_485000_7611000_1K_Las.zip',      2013),
    ('Lower_Balonne_2018_Prj_SW_623000_6856000_1K_Las.zip',         2018),
    ('Laura22021-C3-AHD_2118270_55_0001_0001.laz',                  2021),
    ('Herbert1Lidar2020-C3-AHD_3227992_55_0001_0001.laz',           2020),
    (None,                                                           None),
]
for name, expected in examples:
    got = extract_year(name)
    status = 'OK' if got == expected else f'FAIL (expected {expected})'
    print(f'  {status}  {got}  {str(name)[:50]}')

# %%
gdf['year'] = gdf['name'].apply(extract_year)
print(gdf['year'].value_counts().sort_index())

# %% [markdown]
# ## 4. Group by tile: most recent year + bounding box
#
# Each tile may be covered by several LiDAR polygons, so we optimistically take the most recent year.

# %%
rows = []
for filename, group in gdf.groupby('filename'):
    years = group['year'].dropna()
    year = int(years.max()) if len(years) else None
    geom = unary_union(group.geometry).envelope
    rows.append({'filename': filename, 'year': year, 'geometry': geom})

result = gpd.GeoDataFrame(rows, crs=gdf.crs).to_crs('EPSG:3857')
print(f'{len(result)} tiles, year range {result["year"].min()}–{result["year"].max()}')
print(result['year'].value_counts().sort_index())

# %% [markdown]
# ## 5. Save

# %%
result.to_file('tiff_footprints_years.gpkg', layer='tiff_footprints_years', driver='GPKG')
print('Saved tiff_footprints_years.gpkg')
