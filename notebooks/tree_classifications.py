# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Demo-ing the classification pipeline
import os

# %%
# %%time
# Example 1: Pre-classified lidar and binary outputs
from shelterbelts.classifications.lidar import lidar
laz_file = '../data/Young201709-LID1-C3-AHD_6306194_55_0002_0002.laz'
outdir = '../outdir'

stub = 'pre-classified-trees'
category5 = True  # This means we use their pre-labelled point classifications for trees > 2m 
binary = True   
resolution = 1   # metres (horizontal)

da1 = lidar(laz_file, outdir, stub, resolution, category5=category5, binary=True)

# %%
from shelterbelts.apis.worldcover import visualise_categories, tif_categorical
from shelterbelts.util.binary_trees import cmap_woody_veg, labels_woody_veg
visualise_categories(da1, None, cmap_woody_veg, labels_woody_veg, "pre-classified binary trees > 2m")

# %%
# %%time
# Example 2: Creating our own canopy height model and percentage tree cover outputs
stub = 'our-trees'
category5 = False  # This means we generate our own canopy height model (instead of using their pre-labelled tree classifications)
binary = False   # This means we calculate percentage tree cover instead of just binary labels
height_threshold = 2  # metres (vertical)
resolution = 10       # metres (horizontal)

da2 = lidar(laz_file, outdir, stub, resolution, height_threshold=height_threshold, category5=category5, binary=binary)

# %%
# Converting to a binary tif with a specific percent cover threshold
percent_cover_threshold = 10

da2_binary = da2 > percent_cover_threshold 
filename = os.path.join(outdir, f'{stub}_{height_threshold}m_{percent_cover_threshold}p.tif')
tif_categorical(da2_binary, filename, cmap_woody_veg)

# %%
visualise_categories(da2_binary, None, cmap_woody_veg, labels_woody_veg, "canopy height model 2m treecover > 30%")

# %%
# %%time
# Downloading sentinel imagery for the same area as this tree tif
from shelterbelts.classifications.sentinel_dea import download_ds2

tree_file = '../outdir/our-trees_2m_10p.tif'
start_date="2020-01-01"
end_date="2021-01-01"
ds2 = download_ds2(tree_file, start_date, end_date, outdir)

# %%
# Example band at a specific timepoint
ds2['nbart_red'].isel(time=0).plot()

# %%
# %%time
# Merging the satellite imagery and tree raster
from shelterbelts.classifications.merge_inputs_outputs import tile_csv

sentinel_file = '../outdir/our-trees_2m_10p_ds2_2020-01-01.pkl'
radius = 4   # Radius of the kernel used for spatial variance
spacing = 1  # Distance between each point chosen for training/testing
df = tile_csv(sentinel_file, tree_file, outdir, radius, spacing)
df


# %%
# %%time
# Training a neural network
from shelterbelts.classifications.neural_network import neural_network

training_file = '../outdir/our-trees_2m_10p_df_r4_s1.csv'
stratification_columns = ['tree_cover']
neural_network(training_file, outdir, stub, output_column='tree_cover', drop_columns=['x', 'y', 'tile_id', 'spatial_ref'], stratification_columns=stratification_columns, epochs=20)


# %%
from shelterbelts.apis.worldcover import visualise_categories, tif_categorical
from shelterbelts.util.binary_trees import cmap_woody_veg, labels_woody_veg
tree_file = '..outdir/our-trees_2m_10p.tif'
outdir = '../outdir'
sentinel_file = '../outdir/our-trees_2m_10p_ds2_2020-01-01.pkl'

# %%
# %%time
# Predicting new locations
from shelterbelts.classifications.predictions_batch import tif_prediction
filename_scaler = '../outdir/our-trees_scaler.pkl'
filename_model = '../outdir/our-trees_nn.keras'
da_predicted = tif_prediction(sentinel_file, outdir, filename_model, filename_scaler)


# %%
# Plotting the predicted trees
visualise_categories(da_predicted, None, cmap_woody_veg, labels_woody_veg, "Predicted tree_cover")

# %%
