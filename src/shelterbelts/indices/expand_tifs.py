import rioxarray as rxr
from shapely.geometry import box
import geopandas as gpd

from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds


def expand_tif(filename, num_pixels=10):
    """Expand the tif by a certain number of pixels, to avoid edge effects when running indices at scale"""


filename = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_148/34_93-148_90_y2024_predicted.tif'

da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

da

bounds = da.rio.bounds()

# +
buffer = 100  # 10m x 10 pixels
minx, miny, maxx, maxy = bounds
expanded_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

# Create a shapely box (polygon)
geom = box(*expanded_bounds)

# Save as gpkg
gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:3857').to_file('/scratch/xe2/cb8590/tmp/expanded_bounds.gpkg')

# -



merged_folder = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'

gpkg = 'subfolders_merged_predicted_footprints.gpkg'

# %%time
gdf = bounding_boxes(merged_folder, crs='EPSG:3857')

gdf = gpd.read_file('/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted/subfolders_merged_predicted_footprints.gpkg')

tmpdir = '/scratch/xe2/cb8590/tmp'
unique_stub = 'TEST'  # Needs to be unique to this tile, or you get merge errors

mosaic, out_meta = merge_tiles_bbox(expanded_bounds, tmpdir, unique_stub, merged_folder, gpkg, 'filename', verbose=False)     # Need to include the DATA... in the stub so we don't get rasterio merge conflicts
ds_expanded = merged_ds(mosaic, out_meta, 'expanded')

ds_expanded['expanded'].rio.to_raster('/scratch/xe2/cb8590/tmp/expanded.tif')
