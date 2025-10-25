import glob
import pickle
import argparse
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

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from dask.distributed import Client


# +
# %%time 
csv_glob = '/scratch/xe2/cb8590/Nick_training_lidar_noyear/g*.csv'
out_parquet = '/scratch/xe2/cb8590/Nick_training_lidar_noyear/df_all_parquet'

ddf = dd.read_csv(csv_glob, assume_missing=True)   # lazy, parallel
ddf.to_parquet(out_parquet, engine='pyarrow', write_index=False)
# no compute() here — Dask does the work when to_parquet runs

# -

ddf = dd.read_parquet(out_parquet, engine='pyarrow')
print(ddf.columns)
print(ddf.npartitions)


df_crs = pd.read_csv('/g/data/xe2/cb8590/Nick_outlines/nick_bbox_year_crs.csv')
df_crs['tile_id'] = df_crs['tif'].str.split('.').str[0]
ddf = ddf.merge(df_crs[['tile_id', 'crs']], on='tile_id', how='left')


# +

def reproject_partition(pdf):
    # pdf is a pandas DataFrame for one partition
    # skip empty partitions
    if pdf.shape[0] == 0:
        return pdf
    # some rows might already be EPSG:4326; here we handle arbitrary crs per-row
    # assume 'crs', 'x', 'y' exist
    xs = []
    ys = []
    for crs, xcol, ycol in zip(pdf['crs'].fillna('EPSG:4326'), pdf['x'], pdf['y']):
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x2, y2 = transformer.transform(xcol, ycol)
        xs.append(x2); ys.append(y2)
    pdf = pdf.copy()
    pdf['x'] = xs
    pdf['y'] = ys
    pdf['crs'] = 'EPSG:4326'
    return pdf

# meta: sketch of output dtypes (adjust types/names to match your dataframe)
meta = ddf._meta.copy()
meta['x'] = float
meta['y'] = float
meta['crs'] = object

ddf = ddf.map_partitions(reproject_partition, meta=meta)


# +
# %%time
# Example aggregation (cheap)
counts = ddf.groupby('tile_id').size().compute()
print(counts.head())

# Persist the transformed dataset back to Parquet
out_parquet_4326 = '/scratch/xe2/cb8590/Nick_training_lidar_noyear/df_all_parquet_4326'
ddf.to_parquet(out_parquet_4326, engine='pyarrow', write_index=False)

# -

# %%time
df_all = dd.read_parquet(out_parquet_4326).compute()   # bring to pandas
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_noyear/df_all_4326.feather')






# +
# # %%time
# # combine_csvs('Nick_training_allyears_s5')
# combine_csvs('Nick_training_lidar_noyear')

# +
csv_folder = 'Nick_training_lidar_noyear'
csv_glob = f'/scratch/xe2/cb8590/{csv_folder}/g*.csv'
files = glob.glob(csv_glob)


# -

files = files[:1000]

# Another option for large datasets
ddf = dd.read_csv(files, assume_missing=True)
ddf.to_parquet('/scratch/xe2/cb8590/tmp/df_all_years_merged_parquet', engine='pyarrow', write_index=False)

