# Needed for running notebooks
# When running tests, it runs from the repo root. 
# When running a notebook locally, it runs from within the notebooks folder. 
# When running a notebook on gadi, it runs from the home directory
from shelterbelts.utils.filepaths import setup_repo_path
setup_repo_path()

from shelterbelts.indices.all_indices import indices_tif
from shelterbelts.utils.filepaths import get_filename

# +
# %%time
stub = 'g2_26729'
test_filename = get_filename(f'{stub}_binary_tree_cover_10m.tiff')

# Mostly default parameters
indices_tif(test_filename)
# -


indices_tif(
    test_filename,
    stub="more-shelterbelts",
    min_patch_size=5,  # Less scattered trees
    edge_size=5,  # Thicker core edges
    distance_threshold=30,  # Larger shelter radius
    density_threshold=3,  # Allows more sheltered areas
    buffer_width=5,  # Thicker riparian & road buffers
    min_shelterbelt_length=10,  # Allows shelterbelts to be shorter
)


indices_tif(
    test_filename,
    stub="less-shelterbelts",
    min_patch_size=30,  # More scattered trees
    distance_threshold=10,  # Smaller shelter radius
    density_threshold=10,  # Forces less sheltered areas
    buffer_width=2,  # Thinner riparian & road buffers
    strict_core_area=False,  # More core areas
    max_shelterbelt_width=4,  # Forces shelterbelts to be thinner
)


