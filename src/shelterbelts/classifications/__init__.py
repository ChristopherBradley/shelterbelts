"""Classifications module - machine learning and data classification tools."""

from .bounding_boxes import bounding_boxes
from .binary_trees import cmap_woody_veg
from .neural_network import my_train_test_split, inputs_outputs_split, class_accuracies_overall
from .merge_inputs_outputs import aggregated_metrics
from .merge_lidar import merge_lidar

__all__ = [
    'bounding_boxes',
    'cmap_woody_veg',
    'my_train_test_split',
    'inputs_outputs_split',
    'class_accuracies_overall',
    'aggregated_metrics',
    'merge_lidar',
]

# Optional import that requires datacube (needs to be run on NCI)
try:
    from .sentinel_nci import download_ds2_bbox
    __all__.append('download_ds2_bbox')
except ImportError:
    pass
