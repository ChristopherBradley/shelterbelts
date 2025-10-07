# Comparing alpha earth accuracy and performance with sentinel


import pandas as pd
from pathlib import Path
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
    limit: int = None
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
# %%time
random_seed=42
n_samples=2000
limit=1000
folder_path="/scratch/xe2/cb8590/alphaearth"
output_path=f"/scratch/xe2/cb8590/alphaearth/random_sample_{limit}x{n_samples}.feather"

df = sample_and_combine_csvs(
    folder_path=folder_path,
    output_path=output_path,
    n_samples=n_samples,
    random_seed=random_seed,
    limit=limit
)

print(f"\nFinal dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# +
# df = pd.read_csv('/scratch/xe2/cb8590/alphaearth/g2_24342_binary_tree_cover_10m_alpha_earth_embeddings.csv')
# -

df = df.drop(columns='Unnamed: 0')

df_sample = df.sample(60000)

df_sample = df_sample.drop(columns='source_file')

filename = '/scratch/xe2/cb8590/tmp/alphaearth_60k.csv'
df_sample.to_csv(filename)

combined_df = df_sample

# +
# Count NaN values per column
nan_counts = combined_df.isna().sum()
print(nan_counts[nan_counts > 0])  # Only show columns with NaNs

# Or as a percentage
nan_percentage = (combined_df.isna().sum() / len(combined_df)) * 100
print(nan_percentage[nan_percentage > 0])
# -

df_sample['emb_0'].min()

# +
import matplotlib.pyplot as plt
import numpy as np

# Option 1: Filter out -inf and NaN values
col_data = df_sample['emb_0']
clean_data = col_data[np.isfinite(col_data)]  # Removes both inf and -inf and NaN

clean_data.hist(bins=50)
plt.title(f'Distribution of emb_0 (n={len(clean_data)})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Check how many problematic values you're excluding
print(f"Total values: {len(col_data)}")
print(f"Finite values: {len(clean_data)}")
print(f"NaN values: {col_data.isna().sum()}")
print(f"-inf values: {(col_data == -np.inf).sum()}")
print(f"+inf values: {(col_data == np.inf).sum()}")

# Option 2: Replace -inf with a specific value before plotting
col_data_replaced = col_data.replace(-np.inf, col_data[np.isfinite(col_data)].min())
col_data_replaced.hist(bins=50)
plt.title('Distribution of emb_0 (with -inf replaced)')
plt.show()

# For describe() that handles inf values
print("\nStats for finite values only:")
print(clean_data.describe())
# -

# Comprehensive check
print("Data Quality Report:")
print(f"Total rows: {len(combined_df)}")
print(f"\nRows with any NaN: {combined_df.isna().any(axis=1).sum()}")
print(f"Rows with any -inf: {(combined_df == -np.inf).any(axis=1).sum()}")
print(f"Rows with any +inf: {(combined_df == np.inf).any(axis=1).sum()}")
print(f"\nColumns with NaN values: {combined_df.isna().any().sum()}")
print(f"Columns with -inf values: {(combined_df == -np.inf).any().sum()}")

# +
import numpy as np

# Find rows that contain -inf in any column
mask = (combined_df == -np.inf).any(axis=1)
rows_with_neg_inf = combined_df[mask]

print(f"Found {len(rows_with_neg_inf)} rows with -inf values")
print(rows_with_neg_inf)

# To see which specific columns have -inf in these rows
print("\nColumns with -inf values in these rows:")
inf_columns = (rows_with_neg_inf == -np.inf).sum()
print(inf_columns[inf_columns > 0])

# If you want to see just a few example rows
print("\nFirst 10 rows with -inf:")
print(rows_with_neg_inf.head(10))

# To identify which column(s) have -inf for each row
print("\nWhich columns contain -inf for each row:")
for idx in rows_with_neg_inf.index[:10]:  # First 10 examples
    cols_with_inf = combined_df.columns[(combined_df.loc[idx] == -np.inf)].tolist()
    print(f"Row {idx}: {cols_with_inf}")
# -

# %%time
df_results = random_forest(filename, outdir="/scratch/xe2/cb8590/tmp", stub="alpha_earth", output_column='tree', drop_columns=[], stratification_columns=['tree'])



