import matplotlib.pyplot as plt
import rioxarray as rxr
from shelterbelts.classifications.binary_trees import canopy_height_trees, cmap_woody_veg, labels_woody_veg
from shelterbelts.utils.visualisation import _plot_canopy_height_on_axis, _plot_categories_on_axis
from shelterbelts.utils.filepaths import get_filename

filename = get_filename('milgadara_1kmx1km_CHM_1m.tif')
da_input = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
ds = canopy_height_trees(filename, savetif=False, plot=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
_plot_canopy_height_on_axis(ax1, da_input, title='Canopy Height Input (m)')
_plot_categories_on_axis(ax2, ds['woody_veg'], cmap_woody_veg, labels_woody_veg, 'Binary Tree Cover', legend_inside=True)
plt.tight_layout()