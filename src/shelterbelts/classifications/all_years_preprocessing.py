import glob
import pickle
import joblib
import geopandas as gpd
import pandas as pd
from tensorflow import keras

# Change directory to this repo. Need to do this when using the DEA environment since I can't just pip install -e .
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

from shelterbelts.classifications.neural_network import my_train_test_split, inputs_outputs_split, class_accuracies_overall # Might need to adjust this to work with the random forst model


# +
# %%time
# Prepping the gdf and glob to choose which files to train on
gdf = gpd.read_file('/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg')
gdf_stubs = ['_'.join(filename.split('_')[:2]) for filename in gdf['filename']]
gdf_years = list(gdf['year'].astype(str).values)

csv_glob = '/scratch/xe2/cb8590/Nick_training_lidar_year/*.csv'
files = glob.glob(csv_glob)
glob_stubs = ['_'.join(file.split('/')[-1].split('_')[:2]) for file in files]
glob_years = [file.split('.')[0][-4:] for file in files]
# -

# %%time
# Matching year
glob_stubyears = [
    f"{stub}_{int(year) + offset}"
    for stub, year in zip(gdf_stubs, gdf_years)
    for offset in [0]
]
glob_stubyears_set = set(glob_stubyears)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in glob_stubyears_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_matching.feather')


# +
# %%time
# All years (expecting this to take 15 mins and need at least 8GB memory)
dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_all_years.feather')

# Only took 8 mins, since I was using medium with 18GB of memory
# -

# %%time
df_all = pd.read_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_all_years.feather')
# 84%, not bad not bad

# +
# Apply the trained model to data from other years
# model_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_random_forest_seed1.pkl'
# scaler_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_scaler.pkl'
# model_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_matching_year_random_forest.pkl'
# scaler_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_matching_year_scaler.pkl'
model_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_random_forest_seed1.pkl'
scaler_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_scaler.pkl'

with open(model_path, 'rb') as file:
    model_rf = pickle.load(file)
scaler = joblib.load(scaler_path)

# +
model_filename = '/g/data/xe2/cb8590/models/nn_fft_89a_92s_85r_86p.keras'
model = keras.models.load_model(model_filename)

scaler_filename = '/g/data/xe2/cb8590/models/scaler_fft_89a_92s_85r_86p.pkl'
scaler = joblib.load(scaler_filename)
# -

# %%time
dfs_by_year = {year: group for year, group in df_all.groupby('year')}


dfs_by_year.keys()

dfs_by_year[2017]

year = 2017

# %%time
for year in dfs_by_year.keys():
    print(f"Working on year: {year}")
    outdir = '/scratch/xe2/cb8590/tmp'
    stub = f'old_model_accuracy_{year}'
    non_input_columns = ['tree_cover', 'spatial_ref', 'y', 'x', 'tile_id', 'year', 'start_date', 'end_date']
    output_column = 'tree_cover'
    df_accuracy = class_accuracies_overall(dfs_by_year[year].sample(100000, random_state=1), model, scaler, outdir, stub, non_input_columns, output_column)
    print(df_accuracy)
    print()
    # Matching year: 0.819, 0.821, 0.820 - compared to 0.838 when testing on just the year of interest. 
    # 2 years: 0.822, 0.822, 0.821 - So just 0.1% better...
    # all years with my best model (89% accuracy) 0.826%


