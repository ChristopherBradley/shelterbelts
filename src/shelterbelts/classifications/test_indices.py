+
import datacube
dc = datacube.Datacube()
-

from shelterbelts.indices.full_pipelines import run_pipeline_tif

tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_93-148_38_y2018_predicted_expanded20.tif'

run_pipeline_tif(tif)


