# +
import os

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
    ds_tree_categories = tree_categories(tree_tif, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=False, plot=False)
    ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, ds=ds_tree_categories, savetif=False, plot=False)
    ds_cover = cover_categories(None, worldcover_tif, outdir=outdir, stub=stub, ds=ds_shelter, savetif=False, plot=False)
    ds_buffer = buffer_categories(None, hydrolines_tif, buffer_width=3, outdir=outdir, stub=stub, ds=ds_cover, savetif=False, plot=False)
    ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False)


# +
# %%time
# Repeat that starting from a full worldcover tile
root = '../../..'
outdir = os.path.join(root,'outdir')
stub = 'full_pipeline_w'
# worldcover_tif = os.path.join(outdir, 'g2_26729_worldcover.tif')
worldcover_tif = '/Users/christopherbradley/Documents/PHD/Data/Worldcover_Australia/ESA_WorldCover_10m_2021_v200_S36E147_Map.tif'
ds_woody_veg = worldcover_trees(worldcover_tif, outdir)

# replace this with the full hydrolines
hydrolines_tif = os.path.join(root,'outdir/g2_26729_hydrolines.tif')

# -

# %%time
ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, ds=ds_woody_veg, save_tif=True, plot=False)


# %%time
ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, ds=ds_tree_categories, savetif=True, plot=False)
# Took 27 mins for a full worldcover tile

# %%time
ds_cover = cover_categories(None, worldcover_tif, outdir=outdir, stub=stub, ds=ds_shelter, savetif=True, plot=False)


# Took over an hour to do this step for a full worldcover tile and still didn't finish. I should just run this on smaller BARRA sized tiles instead. Better for consistency too.
# %%time
ds_buffer = buffer_categories(None, hydrolines_tif, buffer_width=3, outdir=outdir, stub=stub, ds=ds_cover, savetif=True, plot=False)


# %%time
ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_tif=True, save_csv=False, save_labels=False)


# +
# Do that for all the tiles in a folder, with the option to specify a limit to help scaling up
# -


