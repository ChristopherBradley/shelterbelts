# +

import numpy as np
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box      
from scipy.ndimage import binary_dilation

import matplotlib.pyplot as plt

# from shelterbelts.apis.worldcover import worldcover_bbox, tif_categorical
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds

# -

from shelterbelts.indices.full_pipelines import worldcover_dir, hydrolines_gdb, roads_gdb


def opportunities(da_trees, da_roads, da_gullies, da_ridges, da_worldcover, outdir='.', stub='TEST', tmpdir='.', width=3, contour_spacing=30):
    """
    Parameters
    ----------
        da_trees, da_roads, da_gullies, da_ridges: binary xarrays 
        da_worldcover: Int xarray for grass and crop categories
        outdir: The output directory to save the results.
        stub: Prefix for output files. If not specified, then it appends 'categorised' to the original filename.
        road_width: Number of pixels away from the feature that still counts as within the buffer
            - May want different widths for different buffers later
        contour_spacing: Number of pixels between each contour
        
    Returns
    -------
        ds: an xarray with a band 'opportunities', where the integers represent the categories defined in 'opportunity_labels'.
    
    Downloads
    ---------
        opportunities.tif: A tif file of the 'opportunities' band in ds, with colours embedded.
    """
    


# +
stub='TEST'
tmpdir = '/scratch/xe2/cb8590/'
buffer_width=3

percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2024_actnsw_4326/subfolders/lat_34_lon_140/34_13-141_90_y2024_predicted.tif' # Should be fine
da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
da_trees = da_percent > 50
da_trees = da_trees.astype('uint8')

gdf_hydrolines, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent)
da_hydrolines = ds_hydrolines['gullies']
gdf_roads, ds_roads = hydrolines(None, roads_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent, layer='NationalRoads_2025_09')
da_roads = ds_roads['gullies']

gs_bounds = gpd.GeoSeries([box(*da_trees.rio.bounds())], crs=da_trees.rio.crs)
bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
worldcover_stub = f'TEST' # Anything that might be run in parallel needs a unique filename, so we don't get rasterio merge conflicts
mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, worldcover_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
da_worldcover2 = da_worldcover.rio.reproject_match(da_trees) # Should do this within full_pipelines so it doesn't need to happen twice


# +
# Find options for buffered gullies and roads that currently cropland or grassland
grass_crops = (da_worldcover2 == 30) | (da_worldcover2 == 40)

y, x = np.ogrid[-buffer_width:buffer_width+1, -buffer_width:buffer_width+1]
gap_kernel = (x**2 + y**2 <= buffer_width**2)
buffered_gullies = binary_dilation(da_hydrolines.values, structure=gap_kernel)
gully_opportunities = buffered_gullies & grass_crops & ~da_trees

buffered_roads = binary_dilation(da_roads.values, structure=gap_kernel)
road_opportunities = buffered_roads & grass_crops & ~da_trees & ~gully_opportunities  # Prioritising gullies over roads

# Create a layer that combines these opportunities

# -



plt.imshow(road_opportunities)


