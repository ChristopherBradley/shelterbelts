from shelterbelts.indices.full_pipelines import run_pipeline_tif

# tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_93-148_38_y2018_predicted_expanded20.tif'


# +
# # %%time
# # Working
# tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_22_y2018_predicted_expanded20.tif'
# run_pipeline_tif(tif)

# +
# # %%time
# Works
# tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_22_y2018_predicted_expanded20.tif'
# run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20)

# +
# # %%time
# # Works
# tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_26_y2018_predicted_expanded20.tif'
# run_pipeline_tif(tif)
# -

# %%time
# Kernel died previously. Now working.
tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_26_y2018_predicted_expanded20.tif'
run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20)


