# +
import os

import geopandas as gpd

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics

from shelterbelts.util.binary_trees import worldcover_trees

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


filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_4.gpkg'
gdf_barra_bboxs = gpd.read_file(filename)

# Extract the bbox
bbox_polygon = gdf_barra_bboxs.iloc[0]
bbox = bbox_polygon.geometry.bounds

# Create a stub based on the center of this bbox
centroid = bbox_polygon.geometry.centroid
stub = f"{centroid.y:.2f}-{centroid.x:.2f}".replace(".", "_")[1:]


# Create the files we need to run the full pipeline
from shelterbelts.apis.worldcover import worldcover_bbox, worldcover_cmap, tif_categorical
from shelterbelts.apis.hydrolines import hydrolines

outdir = '/scratch/xe2/cb8590/tmp4'

# +
# %%time
# Save a worldcover tif for this bbox with this stub and outdir
# Can replace this later by using the canopy_height.merge_tiles_bbox and identify_relevant_tiles_boox functions. 
# Just need to make a 'tiles_global.geojson' in /scratch/xe2/cb8590/Worldcover_Australia with the same format as the canopy_height one.
da_worldcover = worldcover_bbox(bbox)   # Took 8 seconds to download a 4km x 4km area from the microsoft planetary computer

# Don't need to do this anymore, now that I've added the option to pass the da as an argument directly
# filename = os.path.join(outdir, f"{stub}_worldcover.tif")    
# tif_categorical(da_worldcover, filename, worldcover_cmap)
# -

hydrolines_gdb = '/g/data/xe2/cb8590/Outlines/SurfaceHydrologyLinesRegional.gdb'
gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=".", stub="TEST", da=da_worldcover, savetif=False, save_gpkg=False)

ds_woody_veg = worldcover_trees(None, None, da_worldcover, savetif=False)

ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=True, plot=False, ds=ds_woody_veg)

# %%time
ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, savetif=True, plot=False, ds=ds_tree_categories)

ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=True, plot=False, da_worldcover=da_worldcover)

ds_buffer = buffer_categories(None, None, buffer_width=3, outdir=outdir, stub=stub, savetif=True, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines)

ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False)


