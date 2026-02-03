"""Utilities module for shelterbelts analysis."""

from .visualization import visualise_categories, visualise_canopy_height, visualise_categories_sidebyside
from .geo import transform_bbox, identify_relevant_tiles_bbox
from .processing import merge_tiles_bbox, merged_ds
from .sphinx_helpers import get_example_data, create_test_woody_veg_dataset, get_example_tree_categories_data
from .io import tif_categorical

__all__ = [
    'visualise_categories',
    'visualise_canopy_height',
    'visualise_categories_sidebyside',
    'transform_bbox',
    'identify_relevant_tiles_bbox',
    'merge_tiles_bbox',
    'merged_ds',
    'get_example_data',
    'create_test_woody_veg_dataset',
    'get_example_tree_categories_data',
    'tif_categorical',
]
