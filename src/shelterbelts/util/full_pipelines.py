# +
import os

import geopandas as gpd
import rasterio

# # Change directory to this repo - this should work on gadi or locally via python or jupyter.
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
src_dir = os.path.join(repo_dir, 'src')
os.chdir(src_dir)
sys.path.append(src_dir)
# print(src_dir)


from shelterbelts.classifications.binary_trees import worldcover_trees, canopy_height_trees
from shelterbelts.apis.worldcover import worldcover_bbox
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import canopy_height_bbox, merge_tiles_bbox, merged_ds

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics



# +
worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'
hydrolines_gdb = '/g/data/xe2/cb8590/Outlines/SurfaceHydrologyLinesRegional.gdb'
canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'

def run_pipeline(bbox, outdir, stub):
    """Starting from a bbox, go through the whole pipeline"""

    # Load worldcover, canopy height and hydrolines
    mosaic, out_meta = merge_tiles_bbox(bbox, outdir, stub, worldcover_dir)
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    ds_canopy_height = canopy_height_bbox(bbox, outdir=outdir, stub=stub, tmpdir=canopy_height_dir, save_tif=False, plot=False, footprints_geojson='tiles_global.geojson')
    gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=".", stub="TEST", savetif=False, save_gpkg=False, da=da_worldcover)

    # Use the canopy height for starting trees (replace this with my predictions later)
    ds_woody_veg = canopy_height_trees(None, savetif=False, da=ds_canopy_height['canopy_height'])

    # Rest of the pipeline (play around with parameters and edit the functions more later)
    ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=False, plot=False, ds=ds_woody_veg)
    ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories)
    ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=False, plot=False, da_worldcover=da_worldcover)
    ds_buffer = buffer_categories(None, None, buffer_width=3, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines)
    ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False)



# -

def run_gdf(func, gdf, limit=None):
    """Run the function on every row in the dataframe
    Need to make sure the order of argments in the function are the same as in the dataframe.
    """
    pass


# +
filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_10.gpkg'
outdir = '/scratch/xe2/cb8590/tmp4'
gdf_barra_bboxs = gpd.read_file(filename)

# Choose a bbox
bbox_polygon = gdf_barra_bboxs.iloc[0]
bbox = bbox_polygon['geometry'].bounds

# Create a stub
centroid = bbox_polygon.geometry.centroid
stub = f"{centroid.y:.2f}-{centroid.x:.2f}".replace(".", "_")[1:]

# +
# %%time
for i, row in gdf_barra_bboxs.iterrows():
    bbox = row['geometry'].bounds
    centroid = row['geometry'].centroid
    stub = f"{centroid.y:.2f}-{centroid.x:.2f}".replace(".", "_")[1:]
    run_pipeline(bbox, outdir, stub) 

# Takes about 4 secs per tile
# So should take about an hour per 1000 tiles. 
# -


