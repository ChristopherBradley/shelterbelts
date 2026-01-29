"""Utilities for Sphinx documentation examples."""

from pathlib import Path
import rioxarray as rxr


def create_test_woody_veg_dataset():
    """Create a test woody vegetation dataset for docstring examples.
    
    Returns a ready-to-use xarray Dataset with a 'woody_veg' variable,
    eliminating file I/O from docstring examples.
    
    Returns
    -------
    xarray.Dataset
        Dataset with 'woody_veg' data array (boolean, 100x100 pixels)
    """
    test_file = get_example_data('g2_26729_binary_tree_cover_10m.tiff')
    da_trees = rxr.open_rasterio(test_file).isel(band=0).drop_vars('band')
    return da_trees.to_dataset(name='woody_veg')


def get_example_data(filename):
    """Find example data file for use in Sphinx plot directives.
    
    Tries multiple paths to accommodate both local development and
    Read the Docs environments.
    
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
        Path.cwd().parent / 'data' / filename,
        Path.cwd().parent.parent / 'data' / filename,
        Path('/Users/christopherbradley/repos/PHD/shelterbelts/data') / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        f"Data file '{filename}' not found in any expected location. "
        f"Checked: {[str(p) for p in possible_paths]}"
    )
