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
# # ELVIS Tiling
#

# %%
import os

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

import rioxarray as rxr
from rasterio.features import shapes
from shapely.geometry import shape


# %%
# # NCI Filepaths
# aus_boundaries = '/g/data/xe2/cb8590/Outlines/AUS_2021_AUST_GDA2020.shp'
# state_boundaries = '/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp'
# outdir = '/scratch/xe2/cb8590/lidar/polygons/elvis_inputs/'

# Local filepaths
aus_boundaries = 'AUS_2021_AUST_GDA2020.shp'
state_boundaries = 'STE_2021_AUST_GDA2020.shp'
outdir = '/home/christopher-bradley/repos/shelterbelts/tmpdir/elvis_polygons'

# %%
state = 'New South Wales'
tile_size = 30_000   # 30km × 30km in EPSG:7855
crs = 7855

# %% [markdown]
# ## 1. Clip the state outline and build a regular grid

# %%
gdf_states = gpd.read_file(state_boundaries)
state_gdf = gdf_states.loc[gdf_states['STE_NAME21'] == state].to_crs(crs)

# Slightly widened NSW bounds chosen to align with the 2 km BARRA grid.
minx, miny, maxx, maxy = -85_000, 5_845_000, 1_250_000, 6_870_000

xs = np.arange(minx, maxx, tile_size)
ys = np.arange(miny, maxy, tile_size)
tiles = [box(x, y, x + tile_size, y + tile_size) for y in ys for x in xs]
grid = gpd.GeoDataFrame(geometry=tiles, crs=state_gdf.crs)

# %%
# Keep only tiles that intersect the state polygon.
state_geom = prep(state_gdf.union_all())
grid = grid[grid.geometry.map(state_geom.intersects)].reset_index(drop=True)
print(f'{len(grid)} tiles intersect {state}')

# %% [markdown]
# ## 2. Write one GeoJSON per tile (for ELVIS downloads)
#
# The Elvis interface lets you choose a geojson, so this creates them for you.

# %%
# %%time
outdir_geojsons = os.path.join(outdir, f'geojsons_{tile_size}')
os.makedirs(outdir_geojsons, exist_ok=True)

filenames = []
for idx, row in grid.iterrows():
    centroid = row.geometry.centroid
    cx, cy = int(centroid.x), int(centroid.y)
    stub = f'tile{idx}_{cx}_{cy}'
    filenames.append(stub)
    gpd.GeoDataFrame([row], crs=grid.crs).to_crs(4326).to_file(
        os.path.join(outdir_geojsons, f'{stub}.geojson'),
        driver='GeoJSON',
    )

grid['filename'] = filenames
grid.to_file(os.path.join(outdir, f'tiles_{tile_size}_{state.replace(" ", "_")}.gpkg'),
             driver='GPKG')

# %%
# Download the crop and grazing masks from here: https://www.agriculture.gov.au/abares/aclump/land-use/land-use-of-australia-2010-11-to-2020-21#downloads
# "Land use of Australia 2020–21 agricultural commodities probability grids - raster package (GeoTIFF and supporting files) (ZIP 246 MB)"

barra_bboxs = "/home/christopher-bradley/Documents/PHD/data/Outlines/barra_bboxs_aus.gpkg"
grazing_modified_pastures = "/home/christopher-bradley/Documents/PHD/data/Outlines/NLUM_v7_250_AgProbabilitySurfaces_2020_21_geo_package_20241128/NLUM_v7_probSurf_2021_320_3_GRAZ_NOTIMBSP.tif"
# cropping_winter_cereals = "/home/christopher-bradley/Documents/PHD/data/Outlines/NLUM_v7_250_AgProbabilitySurfaces_2020_21_geo_package_20241128/NLUM_v7_probSurf_2021_331_5_W_CER.tif"


# %%
# Create a copy of barra_bboxs in just the pixels where grazing_modified_pastures is not no data.
da_graz = rxr.open_rasterio(grazing_modified_pastures, masked=True).isel(band=0).squeeze()
valid_mask = (~da_graz.isnull()).astype('uint8').values

# %%
polygons = [
    shape(geom)
    for geom, val in shapes(valid_mask, transform=da_graz.rio.transform())
    if val == 1
]

# %%
# %%time
gdf_valid = gpd.GeoDataFrame(geometry=polygons, crs=da_graz.rio.crs).dissolve()

# %%
# %%time
gdf_bboxs = gpd.read_file(barra_bboxs).to_crs(da_graz.rio.crs)
barra_grazing = gdf_bboxs[gdf_bboxs.intersects(gdf_valid.geometry.iloc[0])].reset_index(drop=True)

# %%
# %%time
print(f'{len(barra_grazing)} / {len(gdf_bboxs)} barra tiles intersect grazing modified pastures')
outpath = barra_bboxs.replace('.gpkg', '_grazing.gpkg')
barra_grazing.to_file(outpath, driver='GPKG')
print(f'Saved: {outpath}')

