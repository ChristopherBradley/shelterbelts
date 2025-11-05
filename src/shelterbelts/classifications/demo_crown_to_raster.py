# +
# Creating an example of how the crowns got converted into a binary tree raster with the code Nick provided
# -

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
import rioxarray as rxr
import rasterio as ras

# Create a sample gpkg (from John's ACT tree crowns, cropped to a small area around lake burly griffin)
filename_gpkg = '/Users/christopherbradley/Documents/PHD/Data/John Tree Crowns/Black_Mountain_Peninsula.gpkg'
gdf = gpd.read_file(filename_gpkg)

# Create a sample tif file of the same area
filename = '/Users/christopherbradley/Documents/PHD/Data/Worldcover_Australia/ESA_WorldCover_10m_2021_v200_S36E147_Map.tif'
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
da_sliced = da.sel(x=slice(149.075753, 149.108797), y=slice(-35.283475, -35.302250))
da_sliced = da_sliced.rio.reproject(gdf.crs)
filename_tif = '/Users/christopherbradley/Documents/PHD/Data/John Tree Crowns/Worldcover_Slice_7855.tif'
da_sliced.rio.to_raster(filename_tif)

# +
# Nick's code for doing the rasterization
with ras.open(filename_tif) as src:
    raster_meta = src.meta.copy()
    raster_crs = src.crs
    raster_transform = src.transform
    raster_shape = (src.height, src.width)
    
gdf = gdf.to_crs(raster_crs)

tree_cover = np.zeros(raster_shape, dtype=np.uint8)
shapes = ((geom, 1) for geom in gdf.geometry)
tree_cover = rasterize(
    shapes,
    out_shape=raster_shape,
    transform=raster_transform,
    fill=0,  
    all_touched=True,  # This ensures pixels are marked if the crown touches them
    dtype=np.uint8
)

output_path = '/Users/christopherbradley/Documents/PHD/Data/John Tree Crowns/rasterized_example.tiff'
binary_meta = raster_meta.copy()
binary_meta.update(
    dtype='uint8',
    count=1,
    nodata=0 
)
with ras.open(output_path, 'w', **binary_meta) as dst:
    dst.write(tree_cover, 1)
print(f"Saved: {output_path}")

