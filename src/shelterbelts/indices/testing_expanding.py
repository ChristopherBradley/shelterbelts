

from shelterbelts.indices.expand_tifs import expand_tifs, expand_tif


# filename = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_148/34_93-148_90_y2024_predicted.tif'
# folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'
filename =  '/scratch/xe2/cb8590/barra_trees_s4_2020_actnsw_4326_weightings_median/subfolders/lat_28_lon_140/29_01-140_98_y2020_predicted.tif'
folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_2020_actnsw_4326_weightings_median/subfolders'
outdir = '/scratch/xe2/cb8590/tmp'


ds = expand_tif(filename, folder_merged, outdir)

filename = '/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/lat_32_lon_148/33_25-149_70_y2020_predicted.tif'
folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders'
outdir = '/scratch/xe2/cb8590/tmp'
gpkg = '/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/barra_trees_s4_aus_4326_weightings_median_2020_subfolders__footprints.gpkg'
ds = expand_tif(filename, folder_merged, outdir, gpkg)


# +

# # %%time
folder_to_expand = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_144'
folder_merged = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/merged_predicted'
outdir = '/scratch/xe2/cb8590/barra_trees_s4_2024/expanded'


# # %%time
expand_tifs(folder_to_expand, folder_merged, outdir, limit=10)
# 7 secs for 10, means about 30 mins per folder

