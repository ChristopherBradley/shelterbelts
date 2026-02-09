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

# Mostly default parameters
run_pipeline_tif(test_filename, min_shelterbelt_length=15)
# -


run_pipeline_tif(
    test_filename,
    stub="more-shelterbelts",
    min_patch_size=5,  # Less scattered trees
    min_core_size=10000,  # Less core areas (no change in this small example)
    edge_size=5,  # Thicker core edges
    distance_threshold=30,  # Larger shelter radius
    density_threshold=3,  # Allows more sheltered areas
    buffer_width=5,  # Thicker riparian & road buffers
    min_shelterbelt_length=10,  # Allows shelterbelts to be shorter
)


run_pipeline_tif(
    test_filename,
    stub="less-shelterbelts",
    min_patch_size=30,  # More scattered trees
    distance_threshold=10,  # Smaller shelter radius
    density_threshold=10,  # Forces less sheltered areas
    buffer_width=2,  # Thinner riparian & road buffers
    strict_core_area=False,  # More core areas
    max_shelterbelt_width=4,  # Forces shelterbelts to be thinner
)
