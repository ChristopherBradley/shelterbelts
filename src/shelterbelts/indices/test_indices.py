import glob

# # !pip install 'jupyterlab-lsp' 'python-lsp-server[all]'
# # !jupyter labextension install @krassowski/jupyterlab-lsp


from shelterbelts.indices.full_pipelines import run_pipeline_tifs, run_pipeline_tif
# %%time
cover_threshold=50
min_patch_size=20
min_core_size=1000
edge_size=10
max_gap_size=1
distance_threshold=10
density_threshold=5 
buffer_width=3
strict_core_area=True
param_stub = ""
wind_method=None
wind_threshold=15
# crop_pixels = 20
crop_pixels = 0
limit = None
# folder = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717840/uint8_percentcover_res10_height2m/'
# outdir = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717840/linear_tifs'
# 
# folder='/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_28_lon_142'
folder = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/subfolders/lat_34_lon_148'
tmpdir = '/scratch/xe2/cb8590/tmp'
outdir=tmpdir

# +
# run_pipeline_tif(percent_tif, outdir=tmpdir, tmpdir=tmpdir, stub=None, 
#                      wind_method=None, wind_threshold=15,
#                      cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
#                      distance_threshold=10, density_threshold=5, buffer_width=3, strict_core_area=True,
#                      crop_pixels=0)
# -

# %%time
# percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/34_09-149_14_y2018_predicted_expanded20.tif'  # Exceeding memory
percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148/35_33-149_46_y2018_predicted_expanded20.tif' # keyboard interrupt
run_pipeline_tif(percent_tif, outdir=tmpdir, tmpdir=tmpdir, cover_threshold=50, crop_pixels=20)

folder = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_34_lon_148'
outdir='/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/linear_tifs_lat_34_lon_148_percentcover'


percent_tifs = glob.glob(f'{folder}/*.tif')
if limit:
    percent_tifs = percent_tifs[:limit]
percent_tifs[:10]

[(i, p) for i, p in enumerate(percent_tifs) if '34_09-149_14' in p]

percent_tifs = percent_tifs[359:]

for i, percent_tif in enumerate(percent_tifs):
    print(f"Working on tif {i}/{len(percent_tifs)}:", percent_tif)
    run_pipeline_tif(percent_tif, outdir, tmpdir, None, wind_method, wind_threshold, cover_threshold, min_patch_size, edge_size, max_gap_size, distance_threshold, density_threshold, buffer_width, strict_core_area, crop_pixels)


