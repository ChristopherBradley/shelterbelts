# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Shelterbelts'
copyright = '2026, Christopher Bradley'
author = 'Christopher Bradley'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # Support for Google/NumPy style docstrings
    'sphinxarg.ext',  # Support for argparse documentation
    'matplotlib.sphinxext.plot_directive',  # Support for inline plots
]
templates_path = ['_templates']
exclude_patterns = []

# Autodoc settings
autodoc_typehints = 'description'
autosummary_generate = True

# Mock heavy dependencies so they don't need to be installed to build the docs
autodoc_mock_imports = [
    'pandas', 'geopandas', 'rioxarray', 'shapely', 'psutil',
    'numpy', 'rasterio', 'xarray', 'pyproj', 'odc', 'pystac_client',
    'planetary_computer', 'scipy', 'requests', 'sklearn', 'skimage',
    'tensorflow', 'seaborn', 'joblib', 'datacube', 'pdal', 'fiona',
    'windrose', 'pysheds', 'owslib', 'osmnx', 'tqdm', 'dask', 'cql2',
    'openpyxl', 'xlsxwriter', 'pyarrow', 'DAESIM_preprocess',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# html_theme = 'furo'

html_static_path = ['_static']

# Plot directive settings - show only high-resolution PNG
plot_formats = [('hires.png', 150)]
plot_html_show_formats = False
