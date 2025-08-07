# +
import os

import geopandas as gpd

from shelterbelts.util.binary_trees import worldcover_trees
from shelterbelts.apis.worldcover import worldcover_bbox
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import canopy_height_bbox

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics

# -

# %%time
def g2_26729_density():
    # Starting from a tree tif, go through all the steps to create the linear categories, without generating any intermediate files
    root = '../../..'
    outdir = os.path.join(root,'outdir')
    stub = 'full_pipeline'
    tree_tif = os.path.join(root,'data/g2_26729_binary_tree_cover_10m.tiff')
    worldcover_tif = os.path.join(root,'outdir/g2_26729_worldcover.tif')
    hydrolines_tif = os.path.join(root,'outdir/g2_26729_hydrolines.tif')

    # # I tried this starting from a full worldcover tif, and even with the simplest shelter metrics it took over an hour and didn't finish, so not a feasible approach. Should stick to small tiles - maybe 4kmx4km BARRA tiles?
    # worldcover_tif = '/Users/christopherbradley/Documents/PHD/Data/Worldcover_Australia/ESA_WorldCover_10m_2021_v200_S36E147_Map.tif'
    # ds_woody_veg = worldcover_trees(worldcover_tif, outdir)
    # ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, ds=ds_woody_veg, save_tif=True, plot=False)
    
    ds_tree_categories = tree_categories(tree_tif, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=False, plot=False)
    ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, ds=ds_tree_categories, savetif=False, plot=False)
    ds_cover = cover_categories(None, worldcover_tif, outdir=outdir, stub=stub, ds=ds_shelter, savetif=False, plot=False)
    ds_buffer = buffer_categories(None, hydrolines_tif, buffer_width=3, outdir=outdir, stub=stub, ds=ds_cover, savetif=False, plot=False)
    ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False)


def run_gdf(func, gdf):
    """Run the function on every row in the dataframe
    Need to make sure the order of argments in the function are the same as in the dataframe.
    """
    pass


# +
filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_4.gpkg'
outdir = '/scratch/xe2/cb8590/tmp4'
tmpdir = '/scratch/xe2/cb8590/Global_Canopy_Height'
worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'

hydrolines_gdb = '/g/data/xe2/cb8590/Outlines/SurfaceHydrologyLinesRegional.gdb'
# -

gdf_barra_bboxs = gpd.read_file(filename)

# Extract the bbox
bbox_polygon = gdf_barra_bboxs.iloc[0]
bbox = bbox_polygon.geometry.bounds

# Create a stub based on the center of this bbox
centroid = bbox_polygon.geometry.centroid
stub = f"{centroid.y:.2f}-{centroid.x:.2f}".replace(".", "_")[1:]




from shelterbelts.apis.canopy_height import merge_tiles_bbox
import numpy as np
import xarray as xr

mosaic, out_meta, out_trans = merge_tiles_bbox(bbox, outdir, stub, worldcover_dir)


mosaic

layer_name = "worldcover"
save_tif = True

# +
# Create coordinates
transform = out_meta['transform']
height, width = mosaic.shape[1:]
x = np.arange(width) * transform.a + transform.c
y = np.arange(height) * transform.e + transform.f
if transform.e < 0:
    y = y[::-1]

# Create xarray
da = xr.DataArray(
    mosaic,
    dims=("band", "longitude", "latitude"),
    coords={"band": ["band1"], "longitude": y, "latitude": x},
    name=layer_name
).rio.write_crs(out_meta['crs'])
ds = da.to_dataset().squeeze('band').drop_vars(['band'])

if save_tif:
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    output_tiff_filename = os.path.join(outdir, f'{stub}_{layer_name}.tif')
    with rasterio.open(output_tiff_filename, "w", **out_meta) as dest:
        dest.write(mosaic)
    print("Saved:", output_tiff_filename)
# -





# %%time
# Save a worldcover tif for this bbox with this stub and outdir
# Can replace this later by using the canopy_height.merge_tiles_bbox and identify_relevant_tiles_bbox functions. 
# Just need to make a 'tiles_global.geojson' in /scratch/xe2/cb8590/Worldcover_Australia with the same format as the canopy_height one.
da_worldcover = worldcover_bbox(bbox)   # Took 8 seconds to download a 4km x 4km area from the microsoft planetary computer


gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=".", stub="TEST", da=da_worldcover, savetif=False, save_gpkg=False)

# %%time
ds_canopy_height = canopy_height_bbox(bbox, outdir=outdir, stub=stub, tmpdir=tmpdir, save_tif=False, plot=False, footprints_geojson='tiles_global.geojson')
# Brought this down from 12 secs to 3 secs by using a smaller footprints file

# +
# %%time
# Rest of the pipeline
ds_woody_veg = worldcover_trees(None, None, da_worldcover, savetif=False)
ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=True, plot=False, ds=ds_woody_veg)
ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, savetif=True, plot=False, ds=ds_tree_categories)
ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=True, plot=False, da_worldcover=da_worldcover)
ds_buffer = buffer_categories(None, None, buffer_width=3, outdir=outdir, stub=stub, savetif=True, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines)
ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False)

# All this takes less than 1 second, so could have a 10x speed up by loading worldcover from file instead of from planetary computer. 
# Also need to load from file if I want to use the normal or express gadi queues. So let's set that up.
