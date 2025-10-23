import glob
import geopandas as gpd
import pandas as pd

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

# Might be easier to just merge all the files at once instead of just the relevant ones. Should only take 15 mins and 8GB memory.
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
# All years (expecting this to take 15 mins and need at least 8GB memory)
dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_feather('/scratch/xe2/cb8590/Nick_training_lidar_year/df_all_years.feather')



