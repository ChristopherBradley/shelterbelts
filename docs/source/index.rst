Shelterbelts
============

.. image:: https://badge.fury.io/py/shelterbelts.svg
   :target: https://pypi.org/project/shelterbelts/

.. image:: https://img.shields.io/github/license/ChristopherBradley/shelterbelts.svg
   :target: https://github.com/ChristopherBradley/shelterbelts/blob/main/LICENSE

|

This is an open-source Python package for mapping and categorising
shelterbelts (windbreaks) across Australia using satellite imagery, in preparation
for measuring their impacts on agricultural productivity at scale.

Key Features
------------

1. **Tree categorisation** — classifies pixels as scattered trees, patch core, patch edge, or corridors based on nearby connectivity.
2. **Shelter categorisation** — determine sheltered vs. unsheltered areas based on tree density or wind direction, similar to Stewarts et al.
3. **Cover categorisation** — integrates `ESA WorldCover 2021 <https://esa-worldcover.org/>`_ land-cover classes (grassland, cropland, urban, water) with shelter categories
4. **Buffer categorisation** — identify riparian and roadside tree buffers using the `National Surface Hydrology Lines <https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/83107>`_ and `National Roads <https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147684>`_ datasets
5. **Shelter metrics** — compute patch and class landscape statistics similar to `FragStats <https://fragstats.org/index.php/documentation>`_
6. **Opportunities mapping** — identify locations where additional tree planting would provide the greatest shelter benefit
7. **API integrations** — download data from `ANU BARRA-C2 <https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f2551_3726_7908_8861>`_ (wind), `WRI Canopy Height <https://registry.opendata.aws/dataforgood-fb-forestsv2/>`_, and `ESA WorldCover <https://esa-worldcover.org/>`_
8. **Command-line interface** — all index modules can be run in python scripts or directly from the terminal
9. **Scalable** — designed for national-scale processing on HPC systems (NCI Gadi)

Example Output
--------------

The plot below shows the full categorisation pipeline: a binary tree-cover
raster on the left, and the final buffer categories (with gullies, roads, and
ridges identified) on the right.

.. plot::

   import matplotlib.pyplot as plt
   import rioxarray as rxr
   from shelterbelts.indices.buffer_categories import buffer_categories, buffer_categories_cmap, buffer_categories_labels
   from shelterbelts.utils.visualisation import _plot_categories_on_axis
   from shelterbelts.utils.filepaths import get_filename

   # Binary tree cover (left)
   tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
   da_trees = rxr.open_rasterio(tree_file).squeeze('band').drop_vars('band')
   tree_cmap = {0: (255, 255, 255), 1: (14, 138, 0)}
   tree_labels = {0: 'No Trees', 1: 'Woody Vegetation'}

   # Buffer categories with gullies + roads + ridges (right)
   cover_file = get_filename('g2_26729_cover_categories.tif')
   gullies_file = get_filename('g2_26729_hydrolines.tif')
   ridges_file = get_filename('g2_26729_DEM-S_ridges.tif')
   roads_file = get_filename('g2_26729_roads.tif')
   ds_all = buffer_categories(
       cover_file, gullies_file, ridges_data=ridges_file, roads_data=roads_file,
       outdir='/tmp', stub='demo', plot=False, savetif=False
   )

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
   _plot_categories_on_axis(ax1, da_trees, tree_cmap, tree_labels, 'Example Input', legend_inside=True)
   _plot_categories_on_axis(ax2, ds_all['buffer_categories'], buffer_categories_cmap, buffer_categories_labels, 'Example Output', legend_inside=True)
   plt.tight_layout()


Visualise Results
-----------------

You can explore results interactively in the
`Google Earth Engine App <https://christopher-bradley-phd.projects.earthengine.app/view/shelterbelts>`_.


Github Repository
-----------------
You can find installation instructions on the README of the `GitHub repository <https://github.com/ChristopherBradley/shelterbelts>`_.


Parameter Reference
-------------------

The main parameters for categorising shelterbelts are:

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10 50

   * - Parameter
     - Default
     - Low
     - High
     - Description
   * - ``min_patch_size``
     - 20
     - 10
     - 30
     - Minimum area (pixels) to classify as a patch rather than scattered trees
   * - ``min_core_size``
     - 1000
     - 100
     - 10000
     - Minimum patch size (pixels) to classify as a core area
   * - ``edge_size``
     - 3
     - 1
     - 5
     - Distance (pixels) defining the edge region around patch cores
   * - ``max_gap_size``
     - 1
     - 0
     - 2
     - Maximum gap (pixels) to bridge when connecting tree clusters
   * - ``buffer_width``
     - 3
     - 1
     - 5
     - Number of pixels away from a feature that still counts as within the buffer
   * - ``distance_threshold``
     - 20
     - 10
     - 30
     - Distance from trees that counts as sheltered
   * - ``density_threshold``
     - 5
     - 3
     - 10
     - Percentage tree cover within ``distance_threshold`` that counts as sheltered
   * - ``wind_threshold``
     - 20
     - 10
     - 30
     - Wind speed threshold in km/h
   * - ``wind_method``
     - WINDWARD
     - MOST_COMMON
     - ALL
     - Method to determine primary wind direction
   * - ``strict_core_area``
     - True
     - False
     - True
     - Whether to enforce strict connectivity for core areas
   * - ``min_shelterbelt_length``
     - 15
     - 10
     - 30
     - Minimum skeleton length (pixels) to classify a cluster as linear
   * - ``max_shelterbelt_width``
     - 6
     - 4
     - 8
     - Maximum skeleton width (pixels) to classify a cluster as linear


API Reference
-------------

.. toctree::
   :maxdepth: 2

   modules
