# +
# Cropping binary tif for Margaret Monaro region
# -

import rioxarray as rxr
import rasterio
import geopandas as gpd

from rioxarray.merge import merge_arrays

from shelterbelts.apis.worldcover import tif_categorical

# +
# file1 = '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/subfolders/lat_34_lon_148_merged_predicted.tif'
# file2 = '/scratch/xe2/cb8590/barra_trees_s4_aus_noxy_df_4326_2020/subfolders/lat_36_lon_148_merged_predicted.tif'

file1 = '/scratch/xe2/cb8590/barra_trees_s4_2024_actnsw_4326/subfolders/lat_34_lon_148_merged_predicted_subfolders_2024.tif'
file2 = '/scratch/xe2/cb8590/barra_trees_s4_2024_actnsw_4326/subfolders/lat_36_lon_148_merged_predicted_subfolders_2024.tif'
# -

da1 = rxr.open_rasterio(file1).isel(band=0).drop_vars('band')
da2 = rxr.open_rasterio(file2).isel(band=0).drop_vars('band')

da1_binary = (da1 > 50).astype('uint8')
da2_binary = (da2 > 50).astype('uint8')

merged = merge_arrays([da1_binary, da2_binary])  # Could I be using this method in my canopy_height.merge_tiles_bbox instead?

gdf = gpd.read_file('/g/data/xe2/cb8590/Outlines/monaro/SMRC_LGA_area.shp')
gdf = gdf.to_crs(merged.rio.crs)

# %%time
clipped = merged.rio.clip(
    gdf.geometry,
    gdf.crs,
    drop=True
)

# %%time
out_tif = "/scratch/xe2/cb8590/tmp/monaro_trees_2024_clipped.tif"
clipped.rio.to_raster(
    out_tif,
    dtype="uint8",
    nodata=0,
    compress="LZW"
)
with rasterio.open(out_tif, "r+") as dst:
    dst.nodata = 0
    dst.write_colormap(
        1,  # band number
        {
            1: (0, 255, 0), # Green
        }
    )
