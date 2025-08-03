# Make this a .ipynb once happy with it.

# Import for visualisation inside the notebook
from shelterbelts.apis.worldcover import visualise_categories

# Tree categories
from shelterbelts.indices.tree_categories import tree_categories, tree_categories_cmap, tree_categories_labels

# +
tree_tif = '../data/g2_26729_binary_tree_cover_10m.tiff'

# Optional parameters
outdir = '../outdir'
stub = 'shelter_indices'
min_patch_size=20  # Will be replaced by min_length once I have the patch metrics
edge_size=3
max_gap_size=1

ds = tree_categories(tree_tif, outdir, stub, min_patch_size, edge_size, max_gap_size)
# -


visualise_categories(ds['tree_categories'], None, tree_categories_cmap, tree_categories_labels, "Tree Categories")

# +
# shelter categories
from shelterbelts.indices.shelter_categories import shelter_categories, shelter_categories_cmap, shelter_categories_labels

category_tif = '../outdir/shelter_indices_categorised.tif'

# Method 1: percentage tree cover
ds_density = shelter_categories(category_tif, outdir=outdir, stub='tree_density', distance_threshold=10, density_threshold=10)

# Method 2: single wind direction
wind_ds = '../outdir/g2_26729_barra_daily.nc'
ds_single = shelter_categories(category_tif, wind_ds, outdir=outdir, stub=stub, wind_method='MOST_COMMON', distance_threshold=10, wind_threshold=15)

# Method 3: Multiple wind directions
wind_ds = '../outdir/g2_26729_barra_daily.nc'
height_tif = '../outdir/g2_26729_canopy_height.tif'
ds_multiple = shelter_categories(category_tif, wind_ds, height_tif, outdir=outdir, stub='multiple_directions', wind_method='ANY', distance_threshold=10)
# -

visualise_categories(ds_single['shelter_categories'], None, shelter_categories_cmap, shelter_categories_labels, "Shelter Categories")

# +
# worldcover categories
from shelterbelts.indices.cover_categories import cover_categories, cover_categories_cmap, cover_categories_labels

shelter_tif = '../outdir/shelter_indices_shelter_categories.tif'
worldcover_tif = '../outdir/g2_26729_worldcover.tif'

ds_cover = cover_categories(shelter_tif, worldcover_tif, outdir=outdir, stub=stub)
# -

visualise_categories(ds_cover['cover_categories'], None, cover_categories_cmap, cover_categories_labels, "Cover Categories")

# +
# buffer categories
from shelterbelts.indices.buffer_categories import buffer_categories, buffer_categories_cmap, buffer_categories_labels

cover_tif = '../outdir/shelter_indices_cover_categories.tif'

# Just riparian buffers
hydrolines_tif = '../outdir/g2_26729_hydrolines.tif'
ds_hydrolines = buffer_categories(cover_tif, hydrolines_tif, outdir=outdir, stub='hydrolines', buffer_width=3)

# Riparian buffers and roads
roads_tif = '../outdir/g2_26729_roads.tif'
ds_roads = buffer_categories(cover_tif, hydrolines_tif, roads_tif=roads_tif, outdir=outdir, stub=stub, buffer_width=3)

# Riparian buffers and catchment boundaries (ridges)
gullies_tif = '../outdir/g2_26729_5m_gullies.tif'
ridges_tif = '../outdir/g2_26729_5m_ridges.tif'
ds_ridges = buffer_categories(cover_tif, gullies_tif, ridges_tif, outdir=outdir, stub='ridges', buffer_width=3)
# -

visualise_categories(ds_roads['buffer_categories'], None, buffer_categories_cmap, buffer_categories_labels, "Buffer Categories")
