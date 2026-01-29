"""Utilities for Sphinx documentation examples."""

from pathlib import Path


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
