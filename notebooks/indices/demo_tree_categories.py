# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tree Categories Demo
#
# Demonstrates the `tree_categories` function with different parameter values.

# %% [markdown]
# ## Setup

# %%
import rioxarray as rxr

from shelterbelts.utils.visualisation import visualise_categories_sidebyside, visualise_categories
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.tree_categories import tree_categories_cmap, tree_categories_labels

# Load example data
binary_tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
da_trees = rxr.open_rasterio(binary_tree_file).isel(band=0).drop_vars('band')
ds_input = da_trees.to_dataset(name='woody_veg')
print(f"Input dimensions: {ds_input['woody_veg'].shape}")

# %% [markdown]
# ## Default Parameters

# %%
ds_default = tree_categories(ds_input, stub='default')
ds_default

# %%
visualise_categories(
    ds_default['tree_categories'],
    colormap=tree_categories_cmap,
    labels=tree_categories_labels
)

# %% [markdown]
# ## Changing the edge_size
#
# The `edge_size` parameter defines the distance (in pixels) from the edge of a patch.
# Areas beyond this distance from edges are classified as "Patch Core".

# %%
ds_edge1 = tree_categories(ds_input, stub='edge1', outdir='/tmp', plot=False, save_tif=False, edge_size=1)
ds_edge5 = tree_categories(ds_input, stub='edge5', outdir='/tmp', plot=False, save_tif=False, edge_size=5)

visualise_categories_sidebyside(
    ds_edge1['tree_categories'], ds_edge5['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="edge_size=1", title2="edge_size=5"
)

# %% [markdown]
# ## Changing the min_patch_size
#
# The `min_patch_size` parameter sets the minimum area (in pixels) for a cluster to be
# considered a patch rather than scattered trees.

# %%
ds_patch10 = tree_categories(ds_input, stub='patch10', outdir='/tmp', plot=False, save_tif=False, min_patch_size=10)
ds_patch30 = tree_categories(ds_input, stub='patch30', outdir='/tmp', plot=False, save_tif=False, min_patch_size=30)

visualise_categories_sidebyside(
    ds_patch10['tree_categories'], ds_patch30['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="min_patch_size=10", title2="min_patch_size=30"
)

# %% [markdown]
# ## Changing the max_gap_size
#
# The `max_gap_size` parameter determines the maximum gap (in pixels) that can be bridged
# when connecting tree clusters into patches.

# %%
ds_gap0 = tree_categories(ds_input, stub='gap0', outdir='/tmp', plot=False, save_tif=False, max_gap_size=0)
ds_gap2 = tree_categories(ds_input, stub='gap2', outdir='/tmp', plot=False, save_tif=False, max_gap_size=2)

visualise_categories_sidebyside(
    ds_gap0['tree_categories'], ds_gap2['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="max_gap_size=0", title2="max_gap_size=2"
)

# %% [markdown]
# ## Changing the strict_core_area method
#
# The `strict_core_area` parameter changes the method for defining core areas.

# %%
ds_strict_false = tree_categories(ds_input, stub='strict_false', outdir='/tmp', plot=False, save_tif=False, strict_core_area=False)
ds_strict_true = tree_categories(ds_input, stub='strict_true', outdir='/tmp', plot=False, save_tif=False, strict_core_area=True)

visualise_categories_sidebyside(
    ds_strict_false['tree_categories'], ds_strict_true['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="strict_core_area=False", title2="strict_core_area=True"
)

# %% [markdown]
# ## Command Line Interface
# You can also use the function from the command line with the same defaults and parameters.

# %%
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

# %%
# !python shelterbelts/indices/tree_categories.py --help

# %%
# %%time
# !python shelterbelts/indices/tree_categories.py {binary_tree_file} --stub command_line_defaults --outdir ../notebooks/indices

# %%
# !python shelterbelts/indices/tree_categories.py {binary_tree_file} --min_patch_size 40 --min_core_size 100 --edge_size 2 --max_gap_size 2 --no-strict-core-area --stub command_line --outdir ../notebooks/indices

# %% [markdown]
# ### Cleanup
# Remove the output files created by this notebook

# %%
# !rm ../notebooks/indices/*.tif
# !rm ../notebooks/indices/*.png
# !rm ../notebooks/indices/*.xml  # These get generated if you load the tifs in QGIS

