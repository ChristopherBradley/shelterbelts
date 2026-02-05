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

stub = 'g2_26729'
test_filename = get_filename(f'{stub}_binary_tree_cover_10m.tiff')


test_filename

# !ls /Users/christopherbradley/repos/PHD/shelterbelts/data/g2_26729_binary_tree_cover_10m.tiff

# %%time
run_pipeline_tif(test_filename)


