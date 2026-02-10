from shelterbelts.classifications.merge_lidar import merge_lidar


# %%time
base_dir = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/opportunities_lat_32_lon_148'
merge_lidar(base_dir, subdir='', suffix='opportunities.tif', dedup=False, crs="EPSG:3857")


# +
# import rioxarray as rxr 
# da = rxr.open_rasterio('/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/opportunities_lat_32_lon_148/33_01-148_02_y2018_predicted_expanded20_opportunities.tif')
# da.rio.crs

# +
# # # %%time
# stub = 'DATA_722660'
# base_dir = f'/scratch/xe2/cb8590/lidar/{stub}'
# subdir='chm'
# suffix='_percentcover_res10_height2m.tif'
# # suffix='_chm_res1.tif'
# merge_lidar(base_dir, subdir=subdir, suffix=suffix)
# # # # # Took 4 mins first time, 1 min after that.
