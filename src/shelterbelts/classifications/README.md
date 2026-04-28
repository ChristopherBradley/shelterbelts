# shelterbelts.classifications

Tools for creating binary tree-cover rasters.

## Six user journeys
This classifications folder covers use cases 4-6 of this repo.

1. Go straight to the Google Earth Engine App and just look at the output shelter categories.
2. Use the indices.all_indices.indices_latlon to auto-download the global canopy height model and create shelter categories for a specific location.
3. Download your own tree raster from somewhere else and use indices_tif to create the shelter categories.
4. Download a laz file and use the classifications.lidar to create a tree raster.
5. Use sentinel_nci or to download sentinel imagery, and apply the pre-trained neural network to create the tree raster.
6. Use sentinel_nci and a folder of tree tifs to train your own random forest or neural network that predicts tree rasters.

## Module map

| Module | What it does |
|--------|--------------|
| `binary_trees` | Convert ESA WorldCover or Meta Canopy Height tifs to binary tree rasters |
| `bounding_boxes` | Collect footprints for a folder of tifs |
| `lidar` | Convert LAZ point clouds to canopy-height + tree-cover rasters via PDAL |
| `sentinel_nci` | Download Sentinel-2 stacks from the DEA datacube (NCI-only) |
| `merge_inputs_outputs` | Pair Sentinel stacks with tree labels to produce training CSVs |
| `combine_csvs` | Combine lots of training csvs from different tiles into a single .feather file (better compression) |
| `neural_network` | Train a pixel-wise NN on the combined training CSV |
| `random_forest` | Alternative to `neural_network` that's easier to tune but less accurate|
| `cnn` | Another alternative that's theoretically better than a multilayer perceptron, but I didn't manage to make it perform as well  |
| `predictions_batch` | Apply the trained model to unseen sentinel imagery |
| `merge_tifs` | Mosaic a folder of tifs into a single raster |


See [`../../../pbs_scripts/README.md`](../../../pbs_scripts/README.md) for the
NCI PBS scripts that run these modules at scale.
