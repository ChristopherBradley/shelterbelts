# +
import os

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics


# +
# %%time

# Full pipeline without canopy height, wind, roads or ridges. Does require worldcover & hydrolines. 
# Could make a version without worldcover, but it's really easy to download and I think it makes the tifs much prettier.
# Could make a version without hydrolines, but would need dem instead or no analysis of riparian buffers. Also needs a high quality DEM (5m resolution or better), because the 30m DEM's give very poor estimates of riparian buffers from my experience.

# Starting from a tree tif, go through all the steps to create the linear categories, without generating any intermediate files
root = '../../..'
tree_tif = os.path.join(root,'data/g2_26729_binary_tree_cover_10m.tiff')
worldcover_tif = os.path.join(root,'outdir/g2_26729_worldcover.tif')
hydrolines_tif = os.path.join(root,'outdir/g2_26729_hydrolines.tif')
outdir = os.path.join(root,'outdir')
stub = 'full_pipeline'

ds_tree_categories = tree_categories(tree_tif, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=False, plot=False)
ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, ds=ds_tree_categories, savetif=False, plot=False)
ds_cover = cover_categories(None, worldcover_tif, outdir=outdir, stub=stub, ds=ds_shelter, savetif=False, plot=False)
ds_buffer = buffer_categories(None, hydrolines_tif, buffer_width=3, outdir=outdir, stub=stub, ds=ds_cover, savetif=False, plot=False)

ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer)
# -

linear_tif = os.path.join(root,f'outdir/{stub}_linear_categories.tif')
dfs_classes = class_metrics(linear_tif, outdir, stub, ds=ds_linear)

# !ls

# +
# Repeat that starting from a full worldcover tile

# +
# Do that for all the tiles in a folder, with the option to specify a limit to help scaling up
# -


