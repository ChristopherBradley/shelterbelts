API Reference
=============

Indices
=======


Tree Categories

.. autofunction:: shelterbelts.indices.tree_categories.tree_categories


Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: shelterbelts.indices.tree_categories
   :func: parse_arguments
   :prog: python -m shelterbelts.indices.tree_categories

Shelter Categories
------------------

.. autofunction:: shelterbelts.indices.shelter_categories.shelter_categories

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: shelterbelts.indices.shelter_categories
   :func: parse_arguments
   :prog: python -m shelterbelts.indices.shelter_categories

Cover Categories
----------------

.. autofunction:: shelterbelts.indices.cover_categories.cover_categories

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: shelterbelts.indices.cover_categories
   :func: parse_arguments
   :prog: python -m shelterbelts.indices.cover_categories

Buffer Categories
-----------------

.. autofunction:: shelterbelts.indices.buffer_categories.buffer_categories

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: shelterbelts.indices.buffer_categories
   :func: parse_arguments
   :prog: python -m shelterbelts.indices.buffer_categories

Shelter Metrics
---------------

Patch Metrics
~~~~~~~~~~~~~

.. autofunction:: shelterbelts.indices.shelter_metrics.patch_metrics

Class Metrics
~~~~~~~~~~~~~

.. autofunction:: shelterbelts.indices.shelter_metrics.class_metrics

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: shelterbelts.indices.shelter_metrics
   :func: parse_arguments
   :prog: python -m shelterbelts.indices.shelter_metrics

Full Indices Pipeline
----------------------

Run the end-to-end indices pipeline on percent-cover rasters.

.. autofunction:: shelterbelts.indices.all_indices.indices_tif

.. autofunction:: shelterbelts.indices.all_indices.indices_csv

.. autofunction:: shelterbelts.indices.all_indices.indices_tifs

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: shelterbelts.indices.all_indices
   :func: parse_arguments
   :prog: python -m shelterbelts.indices.all_indices

APIs
====

WorldCover
----------

.. autofunction:: shelterbelts.apis.worldcover.worldcover

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: shelterbelts.apis.worldcover
   :func: parse_arguments
   :prog: python -m shelterbelts.apis.worldcover

BARRA Daily
-----------

.. autofunction:: shelterbelts.apis.barra_daily.barra_daily

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: shelterbelts.apis.barra_daily
   :func: parse_arguments
   :prog: python -m shelterbelts.apis.barra_daily
