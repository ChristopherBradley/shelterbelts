import matplotlib.pyplot as plt
import rioxarray as rxr
from shelterbelts.indices.buffer_categories import buffer_categories, buffer_categories_cmap, buffer_categories_labels
from shelterbelts.utils.visualisation import _plot_categories_on_axis
from shelterbelts.utils.filepaths import get_filename

# Binary tree cover (left)
tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
da_trees = rxr.open_rasterio(tree_file).squeeze('band').drop_vars('band')
tree_cmap = {0: (255, 255, 255), 1: (14, 138, 0)}
tree_labels = {0: 'No Trees', 1: 'Woody Vegetation'}

# Buffer categories with gullies + roads + ridges (right)
cover_file = get_filename('g2_26729_cover_categories.tif')
gullies_file = get_filename('g2_26729_DEM-S_gullies.tif')
ridges_file = get_filename('g2_26729_DEM-S_ridges.tif')
roads_file = get_filename('g2_26729_roads.tif')
ds_all = buffer_categories(
    cover_file, gullies_file, ridges_data=ridges_file, roads_data=roads_file,
    outdir='/tmp', stub='demo', plot=False, savetif=False
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
_plot_categories_on_axis(ax1, da_trees, tree_cmap, tree_labels, 'Example Input', legend_inside=True)
_plot_categories_on_axis(ax2, ds_all['buffer_categories'], buffer_categories_cmap, buffer_categories_labels, 'Example Output', legend_inside=True)
plt.tight_layout()