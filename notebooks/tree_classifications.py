# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Demo-ing the classification pipeline

# %%
# Can take this out when running locally
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
src_dir = os.path.join(repo_dir, 'src')
os.chdir(src_dir)
sys.path.append(src_dir)
# print(src_dir)

# %%
# Calculating percent tree cover from a .laz file. Only have this working locally right now.
# from shelterbelts.classifications.lidar import lidar

# %%
# Prepping filenames
tree_dir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/'
stub = 'g2_26729_binary_tree_cover_10m'
tree_file = os.path.join(tree_dir, f'{stub}.tiff')
outdir = '/scratch/xe2/cb8590/tmp'

start_date="2020-01-01"
end_date="2021-01-01"

# %%
# %%time
# Downloading sentinel imagery - will replace with the local version
from shelterbelts.classifications.sentinel_nci import download_ds2
ds2 = download_ds2(tree_file, start_date, end_date, outdir=outdir)


# %%
# Example band at a specific timepoint
ds2['nbart_red'].isel(time=0).plot()

# %%
# %%time
# Merging satellite imagery and the tree raster
from shelterbelts.classifications.merge_inputs_outputs import tile_csv
# sentinel_file = '/scratch/xe2/cb8590/tmp/g2_26729_binary_tree_cover_10m_ds2_2020.pkl'
sentinel_file = os.path.join(outdir, f'{stub}_ds2_{start_date[:4]}.pkl')  # Probably better to just use the full filepath in the final example notebook
df = tile_csv(sentinel_file, tree_file=tree_file, outdir=outdir, radius=4, spacing=1)
df


# %%
# %%time
# Training a neural network
from shelterbelts.classifications.neural_network import neural_network
training_file = '/scratch/xe2/cb8590/tmp/g2_26729_binary_tree_cover_10m_df_r4_s1.csv'
drop_columns = ['x', 'y', 'tile_id', 'spatial_ref']
neural_network(training_file, outdir, stub, output_column='tree_cover', drop_columns=drop_columns, stratification_columns=['tree_cover'], epochs=20)



# %%
# %%time
# Predicting new locations
from shelterbelts.classifications.predictions_batch import tif_prediction
filename_scaler = '/scratch/xe2/cb8590/tmp/g2_26729_binary_tree_cover_10m_scaler.pkl'
filename_model = '/scratch/xe2/cb8590/tmp/g2_26729_binary_tree_cover_10m_nn.keras'
ds = tif_prediction(sentinel_file, outdir, filename_model, filename_scaler)
ds


# %%
# Plotting the predicted trees
ds.plot()
