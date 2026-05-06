API Reference
=============


WorldCover
----------

.. autofunction:: shelterbelts.apis.worldcover.worldcover

BARRA Daily
-----------

.. autofunction:: shelterbelts.apis.barra_daily.barra_daily

Canopy Height
-------------

.. autofunction:: shelterbelts.apis.canopy_height.canopy_height


Indices
=======


Tree Categories
---------------

.. autofunction:: shelterbelts.indices.tree_categories.tree_categories

Shelter Categories
------------------

.. autofunction:: shelterbelts.indices.shelter_categories.shelter_categories

Cover Categories
----------------

.. autofunction:: shelterbelts.indices.cover_categories.cover_categories

Buffer Categories
-----------------

.. autofunction:: shelterbelts.indices.buffer_categories.buffer_categories

Shelter Metrics
---------------

Patch Metrics
~~~~~~~~~~~~~

.. autofunction:: shelterbelts.indices.shelter_metrics.patch_metrics

Class Metrics
~~~~~~~~~~~~~

.. autofunction:: shelterbelts.indices.shelter_metrics.class_metrics

Full Indices Pipeline
----------------------

Run the end-to-end indices pipeline on a single, or multiple percent-cover rasters.

.. autofunction:: shelterbelts.indices.all_indices.indices_tif

.. autofunction:: shelterbelts.indices.all_indices.indices_csv

.. autofunction:: shelterbelts.indices.all_indices.indices_tifs

.. autofunction:: shelterbelts.indices.all_indices.indices_latlon

Catchments
----------

.. autofunction:: shelterbelts.indices.catchments.catchments

Opportunities
-------------

.. autofunction:: shelterbelts.indices.opportunities.opportunities


Classifications
===============

Tools for turning raw remote-sensing inputs into the binary tree-cover rasters 
that feed the indices pipeline.

Binary Tree Rasters
-------------------

.. autofunction:: shelterbelts.classifications.binary_trees.worldcover_trees

.. autofunction:: shelterbelts.classifications.binary_trees.canopy_height_trees

Bounding Boxes
--------------

.. autofunction:: shelterbelts.classifications.bounding_boxes.bounding_boxes

LiDAR
-----

.. autofunction:: shelterbelts.classifications.lidar.lidar

.. autofunction:: shelterbelts.classifications.lidar.lidar_folder

Sentinel Download
----------------------------

.. autofunction:: shelterbelts.classifications.sentinel_nci.download_ds2_bbox

.. autofunction:: shelterbelts.classifications.sentinel_dea.download_ds2_bbox


Training Pipeline
-----------------

.. autofunction:: shelterbelts.classifications.merge_inputs_outputs.merge_inputs_outputs

.. autofunction:: shelterbelts.classifications.combine_csvs.combine_csvs

.. autofunction:: shelterbelts.classifications.random_forest.random_forest

.. autofunction:: shelterbelts.classifications.neural_network.train_neural_network


Prediction
----------

.. autofunction:: shelterbelts.classifications.predictions.predictions

Mosaicking
----------

.. autofunction:: shelterbelts.classifications.merge_tifs.merge_tifs

