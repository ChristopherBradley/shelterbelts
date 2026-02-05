import glob
import pickle
import argparse
import joblib
import geopandas as gpd
import pandas as pd
from tensorflow import keras
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


def combine_csvs(csv_folder, limit=None):
    """Merge all of the csv files into a single .feather file and attach the coordinates in EPSG:4326 and koppen category"""
    # csv_folder = 'Nick_training_allyears_s5'
    csv_glob = f'/scratch/xe2/cb8590/{csv_folder}/g*.csv'
    files = glob.glob(csv_glob)
    
    limit_stub = ""
    if limit:
        files = files[:limit]
        limit_stub = "_" + str(limit)
    
    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    filename = f'/scratch/xe2/cb8590/{csv_folder}/df_all{limit_stub}.feather'
    df_all.to_feather(filename)
    print(filename)

    df_crs = pd.read_csv('/g/data/xe2/cb8590/Nick_outlines/nick_bbox_year_crs.csv')
    df_crs['tile_id'] = [row['tif'].split('.')[0] for i, row in df_crs.iterrows()]
    df_all = df_all.merge(df_crs[['tile_id', 'crs']])

    # Convert to epsg4326
    dfs = []
    for crs, group in df_all.groupby('crs'):
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x2, y2 = transformer.transform(group['x'].values, group['y'].values)
        group = group.copy()
        group['x'], group['y'], group['crs'] = x2, y2, 'EPSG:4326'
        dfs.append(group)

    df_all = pd.concat(dfs, ignore_index=True)
    filename = f'/scratch/xe2/cb8590/{csv_folder}/df_all_4326{limit_stub}.feather'
    df_all.to_feather(filename)
    print(filename)
    
    # Attach koppen categories
    gdf_koppen = gpd.read_file('/g/data/xe2/cb8590/Outlines/Koppen_Australia_cleaned.gpkg')
    gdf_points = gpd.GeoDataFrame(
        df_all,
        geometry=gpd.points_from_xy(df_all['x'], df_all['y']),
        crs="EPSG:4326"
    )
    gdf_joined = gpd.sjoin(gdf_points, gdf_koppen, how='left', predicate='within')
    df_all['Koppen'] = gdf_joined['Name'].values
    filename = f'/scratch/xe2/cb8590/{csv_folder}/df_all_koppen.feather'
    df_all.to_feather(filename)
    print(filename)

    # Save 1 file per koppen category
    koppen_classes = df_all['Koppen'].unique()
    for koppen_class in koppen_classes:
        df_class = df_all[df_all['Koppen'] == koppen_class]
        filename = f'/scratch/xe2/cb8590/{csv_folder}/df_4326_{koppen_class}{limit_stub}.feather'
        df_class.to_feather(filename)
        print(filename)
        
    return df_all


def evaluate_by_category():
    # Random forest loading
    model_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_random_forest_seed1.pkl'
    scaler_path = '/scratch/xe2/cb8590/Nick_training_lidar_year/RF_df_previous_2years_scaler.pkl'
    with open(model_path, 'rb') as file:
        model_rf = pickle.load(file)
    scaler = joblib.load(scaler_path)

    # Neural network loading
    model_filename = '/g/data/xe2/cb8590/models/nn_df_4326_xy_4326.keras'
    scaler_filename = '/g/data/xe2/cb8590/models/scaler_df_4326_xy_4326.pkl'
    model = keras.models.load_model(model_filename)
    scaler = joblib.load(scaler_filename)

    # %%time
    # Evaluating model per category
    dfs_by_year = {year: group for year, group in df_all.groupby('year')}
    dfs_by_koppen = {year: group for year, group in df_all.groupby('Koppen')}
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


def parse_arguments():
    """Parse command line arguments for download_alphaearth_folder() with default values."""
    parser = argparse.ArgumentParser()

    parser.add_argument("folder", help="Folder containing the csv files to combine")
    parser.add_argument('--limit', type=int, default=None, help='Number of rows (default: all)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    combine_csvs(args.folder)

# +
# # !ls /scratch/xe2/cb8590/Nick_training_lidar_year | wc -l

# +
# # %%time
# # combine_csvs('Nick_training_subset')
# df = combine_csvs('Nick_training_allyears_float32_s5', limit=10)
# df

# +
# df = pd.read_feather('/scratch/xe2/cb8590/Nick_training_allyears_s5/df_all_koppen.feather')

# +
# Check on the Koppen categories
# df = df.drop(columns=['spatial_ref', 'y', 'x', 'start_date', 'end_date', 'crs', 'Koppen'])# df = df.drop(columns=['spatial_ref', 'y', 'x', 'start_date', 'end_date', 'crs', 'Koppen'])

# +
# If they don't exist, then create a subset of df to figure out why

# +
# Remove the date columns that I don't need
# -

# Create a separate training dataset with much smaller filesizes and see if I get the same training accuracy (int16 and float16)



