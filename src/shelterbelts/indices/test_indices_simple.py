from shelterbelts.indices.full_pipelines import run_pipeline_tif

# tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_93-148_38_y2018_predicted_expanded20.tif'


# %%time
# Working
tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_26_y2018_predicted_expanded20.tif'
run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20)

# Working
tif = '/scratch/xe2/cb8590/barra_trees_s4_2020_actnsw_4326_weightings_median/expanded/lat_34_lon_148/34_41-148_42_y2020_predicted_expanded20.tif'
run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20)

# Expected to fail, but actually working fine - with or without the strict core area (although this doesn't change anything)
tif = '/scratch/xe2/cb8590/barra_trees_s4_2020_actnsw_4326_weightings_median/expanded/lat_34_lon_148/34_41-148_42_y2020_predicted_expanded20.tif'
run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20, min_core_size=1000)

# Expecting to fail, but still worked just fine again. I think the issue must have been to do with stitching instead.
tif = '/scratch/xe2/cb8590/barra_trees_s4_2020_actnsw_4326_weightings_median/expanded/lat_34_lon_148/34_49-148_30_y2020_predicted_expanded20.tif'
run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20, min_shelterbelt_length=50)


