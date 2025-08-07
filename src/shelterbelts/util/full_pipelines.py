# +
import os

import geopandas as gpd
import rasterio

from shelterbelts.util.binary_trees import worldcover_trees, canopy_height_trees
from shelterbelts.apis.worldcover import worldcover_bbox
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import canopy_height_bbox, merge_tiles_bbox, merged_ds

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

mosaic, out_meta = merge_tiles_bbox(bbox, outdir, stub, worldcover_dir)
ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})


gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=".", stub="TEST", savetif=False, save_gpkg=False, da=da_worldcover)

ds_canopy_height = canopy_height_bbox(bbox, outdir=outdir, stub=stub, tmpdir=tmpdir, save_tif=False, plot=False, footprints_geojson='tiles_global.geojson')

ds_woody_veg = canopy_height_trees(None, savetif=False, da=ds_canopy_height['canopy_height'])

# %%time
# Rest of the pipeline
ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=True, plot=False, ds=ds_woody_veg)
ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, savetif=True, plot=False, ds=ds_tree_categories)
ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=True, plot=False, da_worldcover=da_worldcover)
ds_buffer = buffer_categories(None, None, buffer_width=3, outdir=outdir, stub=stub, savetif=True, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines)
ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False)


