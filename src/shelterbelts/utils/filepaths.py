"""Utilities for file paths and data location management."""

import sys
from pathlib import Path
import os
import rioxarray as rxr


IS_GADI = Path('/scratch').exists()
if IS_GADI:
    # NCI/Gadi file paths
    default_outdir = '/scratch/xe2/cb8590/tmp'
    default_tmpdir = '/scratch/xe2/cb8590/tmp'
    worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'
    worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
    hydrolines_gdb = '/g/data/xe2/cb8590/Outlines/SurfaceHydrologyLinesRegional.gdb'
    roads_gdb = '/g/data/xe2/cb8590/Outlines/2025_09_National_Roads.gdb'
else:
    # Local defaults
    default_outdir = 'outdir'
    default_tmpdir = 'tmp'
    worldcover_dir = 'data'
    worldcover_geojson = 'g2_26729_worldcover_footprints.geojson'
    hydrolines_gdb = 'data/g2_26729_hydrolines_cropped.gpkg'
    roads_gdb = 'data/g2_26729_roads_cropped.gpkg'

# NCI/Gadi file paths - Canopy height data
canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'
canopy_height_geojson = 'tiles_global.geojson'

# Analysis and comparison data
# `tmpdir` should follow the chosen default_tmpdir so local runs don't target /scratch
tmpdir = default_tmpdir
worldcover_folder = '/scratch/xe2/cb8590/Nick_worldcover_reprojected'
my_prediction_dir = '/scratch/xe2/cb8590/barra_trees_s4_aus_4326_weightings_median_2020/subfolders/'
my_prediction_geojson = 'barra_trees_s4_aus_4326_weightings_median_2020_subfolders__footprints.gpkg'
my_prediction_folder = '/scratch/xe2/cb8590/Nick_2020_predicted'
canopy_height_folder = '/scratch/xe2/cb8590/Nick_GCH'

# NCI/Gadi file paths - Reference data
nick_outlines = '/g/data/xe2/cb8590/Nick_outlines'
nick_aus_treecover_10m = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
koppen_australia = '/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned2.gpkg'

# NCI/Gadi file paths - Model data
nn_models_dir = '/g/data/xe2/cb8590/models'

# NCI/Gadi file paths - BARRA bounding boxes and related data
barra_bboxs_dir = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs'
barra_bboxs_full = '/scratch/xe2/cb8590/tmp/barra_bboxs.gpkg'
state_boundaries = '/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp'
aus_boundaries = '/g/data/xe2/cb8590/Outlines/AUS_2021_AUST_GDA2020.shp'
elvis_outputs_dir = '/scratch/xe2/cb8590/lidar/polygons/elvis_inputs/'

# NCI/Gadi file paths - DEM data
nsw_dem_dir = '/g/data/xe2/cb8590/NSW_5m_DEMs_3857'


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
    """Find example data file for use in Sphinx plot directives.
    
    Parameters
    ----------
    filename : str
        Name of the example data file
    
    Returns
    -------
    str
        Path to the example data file
    
    Raises
    ------
    FileNotFoundError
        If the data file cannot be found in any expected location
    """
    possible_paths = [
        Path('data') / filename,
        Path.cwd().parent / 'data' / filename,
        Path.cwd().parent.parent / 'data' / filename]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        f"Data file '{filename}' not found in any expected location. "
        f"Checked: {[str(p) for p in possible_paths]}"
    )


def setup_repo_path(repo_name='shelterbelts'):
    """Set up repository path for notebooks running in different environments.
    
    Detects whether running from repo root, local notebook, or Gadi,
    updates sys.path with src directory, and changes cwd to src directory.
    
    Parameters
    ----------
    repo_name : str, optional
        Name of the repository directory. Default is 'shelterbelts'.
    
    Returns
    -------
    str
        Path to the repository root directory
    """
    cwd = Path.cwd()
    if cwd.name == repo_name: # Tests
        repo_dir = cwd
    elif cwd.parent.name == repo_name: # Notebooks
        repo_dir = cwd.parent
    elif cwd.parent.parent.name == repo_name: # Could make this recursive, but notebooks only go 2 levels deep.
        repo_dir = cwd.parent.parent
    elif (Path.home() / 'Projects' / repo_name).exists():
        repo_dir = Path.home() / 'Projects' / repo_name  # Gadi
    else:
        raise RuntimeError("Could not find repository root")
    
    repo_dir = str(repo_dir)
    src_dir = os.path.join(repo_dir, 'src')
    os.chdir(src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    return repo_dir