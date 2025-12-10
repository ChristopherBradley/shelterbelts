from shelterbelts.indices.shelter_categories import shelter_categories

# # %%time
# # Working
tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_01-149_26_y2018_predicted_expanded20.tif'
# run_pipeline_tif(tif, cover_threshold=50, crop_pixels=20)

# +

# outdir = '../../../outdir/'
# stub = 'g2_26729'
# category_tif = f"{outdir}{stub}_categorised.tif"
# height_tif = f"{outdir}{stub}_canopy_height.tif"
# wind_nc = f"{outdir}{stub}_barra_daily.nc"
# wind_method = 'MOST_COMMON'
# wind_threshold = 15
# distance_threshold = 20
# minimum_height = 1
# wind_dir='E'
# max_distance=20
# density_threshold=10

# -

densities = shelter_categories(tif, outdir='/scratch/xe2/cb8590/tmp', distance_threshold=10)
# da = shelter_categories(category_tif)


