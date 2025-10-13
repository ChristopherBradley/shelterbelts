# Comparing alpha earth accuracy and performance with sentinel


import pandas as pd
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm

# Takes over a minute to import, I should try to reduce the dependencies.
from shelterbelts.classifications.random_forest import random_forest


# +
def sample_and_combine_csvs(
    folder_path: str,
    output_path: str,
    n_samples: int = 200,
    pattern: str = "*.csv",
    random_seed: int = None,
    limit: int = None,
    recent: bool = False
) -> pd.DataFrame:
    """
    Sample n rows from each CSV file in a folder and combine into a single DataFrame.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing CSV files
    output_path : str
        Path where the output feather file will be saved
    n_samples : int, default=200
        Number of rows to randomly sample from each CSV file
    pattern : str, default="*.csv"
        Glob pattern to match CSV files
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with sampled rows from all CSV files
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    folder = Path(folder_path)
    csv_files = list(folder.glob(pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path} matching pattern '{pattern}'")
    print(f"Found {len(csv_files)} CSV files")

    if recent:
        # Filter to just the tiles where the lidar was taken after 2017
        footprints_percent = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/cb8590_Nick_Aus_treecover_10m_footprints.gpkg'
        gdf_percent = gpd.read_file(footprints_percent)
        footprints_years = '/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg'
        gdf_year = gpd.read_file(footprints_years)
        gdf = gdf_percent.merge(gdf_year[['filename', 'year']])
        gdf_recent = gdf[~gdf['bad_tif'] & (gdf['year'] > 2016)] 
        recent_stubs = [stub.split(".")[0] for stub in gdf_recent['filename']]
        matching_csvs = [p for p in csv_files if any(stub in p.name for stub in recent_stubs)]
        csv_files = matching_csvs
    
    # Randomly shuffle the list of CSV files
    random.shuffle(csv_files)
    
    if limit is not None:
        csv_files = csv_files[:limit]
    print(f"Truncate to {len(csv_files)} CSV files")
    
    dfs = []
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            # Read CSV and sample n rows
            df = pd.read_csv(csv_file)
            
            # If file has fewer rows than n_samples, take all rows
            sample_size = min(n_samples, len(df))
            df_sample = df.sample(n=sample_size, random_state=random_seed)
            
            # Add source file column to track origin
            df_sample['source_file'] = csv_file.name
            
            dfs.append(df_sample)
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No data was successfully loaded from any CSV files")
    
    # Combine all dataframes
    print("Combining dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Combined shape: {combined_df.shape}")
    
    # Save as feather
    print(f"Saving to {output_path}...")
    combined_df.to_feather(output_path)
    
    print("Done!")
    return combined_df




# +
# # %%time
# random_seed=42
# n_samples=2000
# limit=1000
# folder_path="/scratch/xe2/cb8590/alphaearth"
# output_path=f"/scratch/xe2/cb8590/alphaearth/random_sample_{limit}x{n_samples}.feather"

# df = sample_and_combine_csvs(
#     folder_path=folder_path,
#     output_path=output_path,
#     n_samples=n_samples,
#     random_seed=random_seed,
#     limit=limit
# )

# print(f"\nFinal dataset shape: {df.shape}")
# print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
# -

df_old = pd.read_feather('/scratch/xe2/cb8590/alphaearth/random_sample_1000x2000.feather')

df = pd.read_feather('/scratch/xe2/cb8590/alphaearth/random_sample_recent_1000x2000.feather')

df = df.drop(columns='Unnamed: 0')

len(df)

# +
# %%time
num_samples = 1000000
if num_samples > len(df):
    num_samples = len(df)
df_sample = df.sample(num_samples)
df_sample = df_sample.drop(columns='source_file')
df_sample = df_sample[(df_sample != -np.inf).all(axis=1)] # Some rows have -inf for every embedding

filename = f'/scratch/xe2/cb8590/tmp/alphaearth_{num_samples}k.csv'
df_sample.to_csv(filename)
# 6 secs for 60k rows
# 30 secs for 300k rows

df_results = random_forest(filename, outdir="/scratch/xe2/cb8590/tmp", stub="alpha_earth", output_column='tree', drop_columns=[], stratification_columns=['tree'])
# 10 secs for 60k rows
# 1 min for 300k rows

df_results

# Using the old lidar data
# 78% for 60k rows
# 80% for 300k rows
# 82% accuracy, 7 mins for 1 million rows

# Using recent lidar (I wasn't expecting these to be worse, but I'm guessing it's to do with the fact that I removed tiles with <10% or >90% tree cover.
# 73%, 20 secs for 60k rows, 
# 76%, 20 secs for 300k rows, 
# 77%, 20 secs for 1mil rows


# -


