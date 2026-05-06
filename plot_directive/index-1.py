import matplotlib.pyplot as plt
import rioxarray as rxr
from shelterbelts.indices.all_indices import indices_tif
from shelterbelts.indices.shelter_metrics import linear_categories_cmap, linear_categories_labels
from shelterbelts.utils.visualisation import _plot_categories_on_axis
from shelterbelts.utils.filepaths import get_filename

tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
da_trees = rxr.open_rasterio(tree_file).squeeze('band').drop_vars('band')
tree_cmap = {0: (255, 255, 255), 1: (14, 138, 0)}
tree_labels = {0: 'No Trees', 1: 'Woody Vegetation'}

ds_linear, _ = indices_tif(tree_file, outdir='/tmp', stub='test')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
_plot_categories_on_axis(ax1, da_trees, tree_cmap, tree_labels, 'Example Input', legend_inside=True)
_plot_categories_on_axis(ax2, ds_linear['linear_categories'], linear_categories_cmap, linear_categories_labels, 'Example Output', legend_inside=True)
plt.tight_layout()