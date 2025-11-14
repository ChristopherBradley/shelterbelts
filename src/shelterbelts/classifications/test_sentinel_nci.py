import datacube


dc = datacube.Datacube(app='sentinel_download')


from shelterbelts.indices.full_pipelines import run_pipeline_tif

tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_93-148_38_y2018_predicted_expanded20.tif'

run_pipeline_tif(tif)

# + endofcell="--"
# # +
# # # %%time
# gdf = bounding_boxes('/g/data/xe2/cb8590/Nick_Aus_treecover_10m', filetype='.tiff')
# # gdf = bounding_boxes("/scratch/xe2/cb8590/Worldcover_Australia")

# # +
# # # %%time
# folder = "/scratch/xe2/cb8590/Worldcover_Australia"
# stub = "Worldcover_Australia"
# outdir = "/scratch/xe2/cb8590/tmp"
# filetype = 'tif'
# crs = None
# pixel_cover_threshold = None
# tif_cover_threshold = None  # Takes 10 secs so long as this is None
# size_threshold = 80
# remove = False

# bounding_boxes(folder)

# Footprints currently aren't working with the .asc files, but centroids are for some reason.
# folder = '/g/data/xe2/cb8590/NSW_5m_DEMs'
# stub = 'NSW_5m_DEMs'
# outdir = "/g/data/xe2/cb8590/Outlines"

# -

# # # %%time
# bounding_boxes(filepath, outdir, stub, filetype='.asc', limit=10)

# # +
# # # %%time
# gdf = bounding_boxes(filepath, outdir, stub, filetype='.asc', crs='EPSG:4326', limit=10)
# gdf.crs

# # +
# size_threshold=80
# tif_cover_threshold=None
# pixel_cover_threshold=None
# remove=False
# filetype='.asc'
# crs=None
# save_centroids=False
# limit=None
# --
