# Needed for running notebooks
import sys, os
from pathlib import Path
cwd = Path.cwd()
if (cwd / 'src').exists():
    repo_dir = cwd
elif (cwd.parent / 'src').exists():
    repo_dir = cwd.parent
else:
    repo_dir = cwd
repo_dir = str(repo_dir)
src_dir = os.path.join(repo_dir, 'src')
os.chdir(repo_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from shelterbelts.indices.full_pipelines import run_pipeline_tif
from shelterbelts.utils.filepaths import get_filename

# +
# %%time
stub = 'g2_26729'
test_filename = get_filename(f'{stub}_binary_tree_cover_10m.tiff')

# Default parameters
run_pipeline_tif(test_filename)


# +
# Low-threshold demo (faster / more permissive)
# -
run_pipeline_tif(
    test_filename,
    cover_threshold=1,
    min_patch_size=10,
    min_core_size=100,
    edge_size=1,
    max_gap_size=0,
    distance_threshold=10,
    density_threshold=3,
    buffer_width=1,
    wind_threshold=10,
    strict_core_area=False,
    min_shelterbelt_length=10,
    max_shelterbelt_width=4,
)


# +
# High-threshold demo (stricter / larger features)
# -
run_pipeline_tif(
    test_filename,
    cover_threshold=1,
    min_patch_size=30,
    min_core_size=10000,
    edge_size=5,
    max_gap_size=2,
    distance_threshold=30,
    density_threshold=10,
    buffer_width=5,
    wind_threshold=30,
    strict_core_area=True,
    min_shelterbelt_length=30,
    max_shelterbelt_width=8,
)
