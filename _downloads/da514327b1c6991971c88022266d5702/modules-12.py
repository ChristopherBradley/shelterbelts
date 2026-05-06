import matplotlib.pyplot as plt
import rioxarray as rxr
from shelterbelts.classifications.binary_trees import worldcover_trees, cmap_woody_veg, labels_woody_veg
from shelterbelts.apis.worldcover import worldcover_cmap, worldcover_labels
from shelterbelts.utils.visualisation import _plot_categories_on_axis
from shelterbelts.utils.filepaths import get_filename

filename = get_filename('g2_26729_worldcover.tif')
da_input = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
ds = worldcover_trees(filename, savetif=False, plot=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
_plot_categories_on_axis(ax1, da_input, worldcover_cmap, worldcover_labels, 'WorldCover Input', legend_inside=True)
_plot_categories_on_axis(ax2, ds['woody_veg'], cmap_woody_veg, labels_woody_veg, 'Binary Tree Cover', legend_inside=True)
plt.tight_layout()