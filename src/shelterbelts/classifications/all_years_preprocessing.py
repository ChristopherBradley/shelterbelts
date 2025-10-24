import glob
import pickle
import joblib
import geopandas as gpd
import pandas as pd
from tensorflow import keras
from shapely.geometry import Point
from pyproj import Transformer


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

csv_glob = '/scratch/xe2/cb8590/Nick_training_lidar_year/g*.csv'
files = glob.glob(csv_glob)
glob_stubs = ['_'.join(file.split('/')[-1].split('_')[:2]) for file in files]
glob_years = [file.split('.')[0][-4:] for file in files]
# -

# %%time
# Matching year
glob_stubyears = [
    f"{stub}_{int(year) + offset}"
    for stub, year in zip(gdf_stubs, gdf_years)
    for offset in [0, -1]
]
glob_stubyears_set = set(glob_stubyears)
subset_files = [f for f, stubyear in zip(files, glob_stubyears) if stubyear in glob_stubyears_set]
dfs = [pd.read_csv(f) for f in subset_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_matching2.feather')


# +
# # Oops, saving the metrics csv files in the same folder messed up regenerating the training files
# # !ls /scratch/xe2/cb8590/Nick_training_lidar_year/*metrics*

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

# %%time
df_matching = pd.read_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_matching_year.feather')
# 84%, not bad not bad

df_matching = df_all

# +
# %%time
# Adding coordinates in EPSG:4326

# Add the crs to each row
df_crs = pd.read_csv('/g/data/xe2/cb8590/Nick_outlines/nick_bbox_year_crs.csv')
df_crs['tile_id'] = [row['tif'].split('.')[0] for i, row in df_crs.iterrows()]
df_matching = df_matching.merge(df_crs[['tile_id', 'crs']])

# Convert to epsg4326
dfs = []
for crs, group in df_matching.groupby('crs'):
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x2, y2 = transformer.transform(group['x'].values, group['y'].values)
    group = group.copy()
    group['x'], group['y'], group['crs'] = x2, y2, 'EPSG:4326'
    dfs.append(group)

df_matching = pd.concat(dfs, ignore_index=True)
df_matching.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_all_4326.feather')
# -

# %%time
df_matching.sample(2000000).to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_2mil_random.feather')

# Add koppen category to each row
gdf_koppen = gpd.read_file('/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned.gpkg')

# +
# %%time
# 1. Convert df_matching to GeoDataFrame
gdf_points = gpd.GeoDataFrame(
    df_matching,
    geometry=gpd.points_from_xy(df_matching['x'], df_matching['y']),
    crs="EPSG:4326"
)
gdf_joined = gpd.sjoin(gdf_points, gdf_koppen, how='left', predicate='within')
df_matching['Koppen'] = gdf_joined['Name'].values

# 5 mins for 4 million points with X-Large compute
# -


df_encoded = pd.get_dummies(df_matching, columns=['Koppen'], prefix='Koppen')


df_encoded.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_all_koppen.feather')

df_matching['Koppen'].value_counts()

# %%time
koppen_classes = df_matching['Koppen'].unique()
for koppen_class in koppen_classes:
    df_class = df_matching[df_matching['Koppen'] == koppen_class]
    filename = f'/scratch/xe2/cb8590/Nick_training_lidar_year/df_4326_{koppen_class}.feather'
    df_class.to_feather(filename)
    print(filename)

len(df_matching)



df_matching['y'].min(), df_matching['y'].max(), df_matching['x'].min(), df_matching['x'].max()

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
# model_filename = '/g/data/xe2/cb8590/models/nn_fft_89a_92s_85r_86p.keras'
# scaler_filename = '/g/data/xe2/cb8590/models/scaler_fft_89a_92s_85r_86p.pkl'
model_filename = '/g/data/xe2/cb8590/models/nn_df_4326_xy_4326.keras'
scaler_filename = '/g/data/xe2/cb8590/models/scaler_df_4326_xy_4326.pkl'


model = keras.models.load_model(model_filename)
scaler = joblib.load(scaler_filename)
# -

# %%time
dfs_by_year = {year: group for year, group in df_all.groupby('year')}
dfs_by_year.keys()


# %%time
dfs_by_koppen = {year: group for year, group in df_matching.groupby('Koppen')}

# %%time
for koppen in dfs_by_koppen.keys():
    print(f"Working on koppen: {koppen}")
    outdir = '/scratch/xe2/cb8590/tmp'
    stub = f'new_model_accuracy_{koppen}'
    non_input_columns = ['tree_cover', 'spatial_ref', 'tile_id', 'year', 'start_date', 'end_date', 'crs', 'Koppen']
    output_column = 'tree_cover'
    df_accuracy = class_accuracies_overall(dfs_by_koppen[koppen].sample(100000, random_state=1), model, scaler, outdir, stub, non_input_columns, output_column)
    print()
    # Matching year: 0.819, 0.821, 0.820 - compared to 0.838 when testing on just the year of interest. 
    # 2 years: 0.822, 0.822, 0.821 - So just 0.1% better...
    # all years with my best model (89% accuracy) 0.826%

outdir = '/scratch/xe2/cb8590/tmp'
stub = f'old_model_accuracy_matching'
non_input_columns = ['tree_cover', 'spatial_ref', 'y', 'x', 'tile_id', 'year', 'start_date', 'end_date']
output_column = 'tree_cover'
df_accuracy = class_accuracies_overall(df_all.sample(100000, random_state=1), model, scaler, outdir, stub, non_input_columns, output_column)
print(df_accuracy)
print()
# Matching year: 0.819, 0.821, 0.820 - compared to 0.838 when testing on just the year of interest. 
# 2 years: 0.822, 0.822, 0.821 - So just 0.1% better...
# all years with my best model (89% accuracy) 0.826%

# +
# %%time
import os
from pathlib import Path

folder = Path("/scratch/xe2/cb8590/alphaearth")

for file in folder.glob("*.csv"):
    new_name = file.with_name(file.stem + "_2020.csv")
    file.rename(new_name)

# -

df = pd.read_csv('/scratch/xe2/cb8590/alphaearth/g1_01001_binary_tree_cover_10m_alpha_earth_embeddings_2020.csv')
df


