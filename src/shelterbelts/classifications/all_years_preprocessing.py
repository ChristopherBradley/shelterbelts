import glob
import pickle
import joblib
import geopandas as gpd
import pandas as pd


from shelterbelts.classifications.neural_network import my_train_test_split, inputs_outputs_split, class_accuracies_overall # Might need to adjust this to work with the random forst model


# +
# %%time
gdf = gpd.read_file('/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg')
gdf_stubs = ['_'.join(filename.split('_')[:2]) for filename in gdf['filename']]
gdf_years = list(gdf['year'].astype(str).values)

csv_glob = '/scratch/xe2/cb8590/Nick_training_lidar_year/*.csv'
files = glob.glob(csv_glob)
glob_stubs = ['_'.join(file.split('/')[-1].split('_')[:2]) for file in files]
glob_years = [file.split('.')[0][-4:] for file in files]

# +
# Matching year
gdf_stubyears = [f"{gdf_stub}_{gdf_year}" for gdf_stub, gdf_year in zip(gdf_stubs, gdf_years)]
glob_stubyears = [f"{glob_stub}_{glob_year}" for glob_stub, glob_year in zip(glob_stubs, glob_years)]
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather(f'/scratch/xe2/cb8590/Nick_training_lidar_year/df_matching_year.feather')

# 3 secs for 100
# 25 secs for 1000
# Expected 2 mins for all 4000

# Might be easier to just merge all the files at once instead of just the relevant ones and then subset after. Should only take 15 mins and 8GB memory. Didn't do it this way because I had launched a session with 4GB memory and didn't want to wait for a new session.
# -

# %%time
# previous_year
gdf_stubyears_prev = [f"{stub}_{int(year) - 1}" for stub, year in zip(gdf_stubs, gdf_years)]
gdf_stubyears_prev_set = set(gdf_stubyears_prev)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_previous_year.feather')


# %%time
# Next year
gdf_stubyears_prev = [f"{stub}_{int(year) + 1}" for stub, year in zip(gdf_stubs, gdf_years)]
gdf_stubyears_prev_set = set(gdf_stubyears_prev)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_next_year.feather')


# %%time
# Previous 2 years
gdf_stubyears_prev = [
    f"{stub}_{int(year) + offset}"
    for stub, year in zip(gdf_stubs, gdf_years)
    for offset in (-1, -2)
]
gdf_stubyears_prev_set = set(gdf_stubyears_prev)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_previous_2years.feather')


# Next 2 years
gdf_stubyears_prev = [
    f"{stub}_{int(year) + offset}"
    for stub, year in zip(gdf_stubs, gdf_years)
    for offset in (1, 2)
]
gdf_stubyears_prev_set = set(gdf_stubyears_prev)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_next_2years.feather')


# %%time
# Previous 3 years
gdf_stubyears_prev = [
    f"{stub}_{int(year) + offset}"
    for stub, year in zip(gdf_stubs, gdf_years)
    for offset in (0, -1, -2)
]
gdf_stubyears_prev_set = set(gdf_stubyears_prev)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_previous_3years.feather')


# %%time
# Previous 4 years
gdf_stubyears_prev = [
    f"{stub}_{int(year) + offset}"
    for stub, year in zip(gdf_stubs, gdf_years)
    for offset in (0, -1, -2, -3)
]
gdf_stubyears_prev_set = set(gdf_stubyears_prev)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_previous_4years.feather')
len(subset_files)

# %%time
# Previous 5 years
gdf_stubyears_prev = [
    f"{stub}_{int(year) + offset}"
    for stub, year in zip(gdf_stubs, gdf_years)
    for offset in (0, -1, -2, -3, -4)
]
gdf_stubyears_prev_set = set(gdf_stubyears_prev)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_previous_5years.feather')
len(subset_files)

# +
# %%time
# All years (expecting this to take 15 mins and need at least 8GB memory)
dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_all_years.feather')

# Only took 8 mins, since I was using medium with 18GB of memory
# -

df_all = pd.read_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_all_years.feather')
# 84%, not bad not bad

# Was going to remove the tiles of a specific year from df_all, so that I'm not predicting tiles it was trained on
glob_stubyears = [f"{stub}_{int(year)}" for stub, year in zip(gdf_stubs, gdf_years)]
glob_stubyears_set = set(glob_stubyears)
subset_files = {f for f, stubyear in zip(files, glob_stubyears) if stubyear in gdf_stubyears_prev_set}
files_set = set(files)
untrained_set = files_set - subset_files
untrained_files = list(untrained_set)

# +
# Apply the trained model to data from other years
model_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_random_forest_seed1.pkl'
scaler_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_scaler.pkl'
# model_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_matching_year_random_forest.pkl'
# scaler_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_matching_year_scaler.pkl'

with open(model_path, 'rb') as file:
    model_rf = pickle.load(file)
scaler = joblib.load(scaler_path)

# +
# %%time
outdir = '/scratch/xe2/cb8590/tmp'
stub = 'matchingyear_apply_to_all_years'
non_input_columns = ['tree_cover', 'spatial_ref', 'y', 'x', 'tile_id', 'year', 'start_date', 'end_date']
output_column = 'tree_cover'
df_accuracy = class_accuracies_overall(df_all.sample(100000, random_state=2), model_rf, scaler, outdir, stub, non_input_columns, output_column)

# Matching year: 0.819, 0.821, 0.820 - compared to 0.838 when testing on just the year of interest. 
# 2 years: 0.822, 0.822, 0.821 - So just 0.1% better...
