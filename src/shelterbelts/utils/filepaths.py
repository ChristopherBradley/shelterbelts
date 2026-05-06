"""Utilities for file paths and data location management."""

from pathlib import Path
import rioxarray as rxr


_repo_root = Path(__file__).resolve().parent.parent.parent.parent

IS_GADI = Path('/scratch').exists()
if IS_GADI:
    # NCI/Gadi file paths
    default_tmpdir = '/scratch/xe2/cb8590/tmp'
    worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'
    hydrolines_gdb = '/g/data/xe2/cb8590/Outlines/SurfaceHydrologyLinesRegional.gdb'
    roads_gdb = '/g/data/xe2/cb8590/Outlines/2025_09_National_Roads.gdb'
else:
    # Local defaults
    default_tmpdir = '.'
    worldcover_dir = str(_repo_root / 'data')  # Using this '/' operator means the filepaths should work on both windows and mac.
    hydrolines_gdb = str(_repo_root / 'data' / 'g2_26729_hydrolines_cropped.gpkg')
    roads_gdb = str(_repo_root / 'data' / 'g2_26729_roads_cropped.gpkg')

# NCI/Gadi - Analysis and comparison data
canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'
canopy_height_geojson = 'tiles_global.geojson'
worldcover_folder = '/scratch/xe2/cb8590/Nick_worldcover_reprojected'
my_prediction_dir = '/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/'
my_prediction_geojson = 'barra_trees_s4_aus_4326_weightings_median_2020_subfolders__footprints.gpkg'
my_prediction_folder = '/scratch/xe2/cb8590/Nick_2020_predicted'
canopy_height_folder = '/scratch/xe2/cb8590/Nick_GCH'
nick_outlines = '/g/data/xe2/cb8590/Nick_outlines'
nick_aus_treecover_10m = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
koppen_australia = '/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned2.gpkg'

# NCI/Gadi file paths - DEM data
nsw_dem_dir = '/g/data/xe2/cb8590/NSW_5m_DEMs_3857'

# Bundled sample fixtures
laz_sample = str(_repo_root / 'data' / 'milgadara_50mx50m.laz')
dem_h_sample = str(_repo_root / 'data' / 'g2_26729_DEM-H.tif')
quartered_tifs_dir = str(_repo_root / 'data' / 'quartered_linear_tifs')
sentinel_sample = str(_repo_root / 'data' / 'g2_019_sentinel_150mx150m.pkl')
training_csv_sample = str(_repo_root / 'data' / 'g2_017_training.csv')

def create_test_woody_veg_dataset():
    """Create a test woody vegetation dataset for docstring examples.
    
    Returns
    -------
    xarray.Dataset
        Dataset with 'woody_veg' data array (boolean, 100x100 pixels)
    """
    test_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
    da_trees = rxr.open_rasterio(test_file).isel(band=0).drop_vars('band')
    return da_trees.to_dataset(name='woody_veg')


def get_example_tree_categories_data():
    """Create tree categories data for docstring examples.
    
    Returns
    -------
    xarray.Dataset
        Dataset with 'tree_categories' data array
    """
    from shelterbelts.indices.tree_categories import tree_categories
    
    ds_input = create_test_woody_veg_dataset()
    ds_cat = tree_categories(ds_input, stub='example', outdir='/tmp', plot=False, save_tif=False)
    return ds_cat


def get_filename(filename):
    """Find example data file bundled in the repository's data/ directory."""
    return str(_repo_root / 'data' / filename)


def get_pretrained_nn(koppen='all'):
    """Return the path to a bundled pre-trained neural network for tree classification."""
    return str(_repo_root / 'models' / f'nn_noxy_df_4326_{koppen}.keras')


def get_pretrained_scaler(koppen='all'):
    """Return the path to the scaler matching a bundled pre-trained neural network."""
    return str(_repo_root / 'models' / f'scaler_noxy_df_4326_{koppen}.pkl')


