# +
# Download the paddock boundaries for all the beta farms

# +
# # !pip install snowflake-connector-python openpyxl
# -

import snowflake.connector
import geopandas as gpd
from shapely import wkt
import pandas as pd

from credentials import password

conn = snowflake.connector.connect(
    user='ANU_CHRISTOPHER',
    password=password,
    account='pawhkoa-mp20061',
    warehouse='FORAGECASTER_WH',
    database='FORAGECASTER_PROD',
    schema='PADDOCK_10_24_2024',
)
cursor = conn.cursor()


# %%time
# Selecting all the columns from beta_paddocks
cursor.execute("""
SELECT FARM_ID, ENCRYPTED_FARM_ID, PADDOCK_ID, ENCRYPTED_PADDOCK_ID, TITLE, CROP_TYPE, PASTURE_STATE, LATITUDE, LONGITUDE, CREATION_DATE, LAST_MODIFIED_DATE, ST_ASWKT(GEOMETRY) AS WKT
FROM FORAGECASTER_PROD.PADDOCK_SOURCE.BETA_PADDOCKS
""")
columns = [col[0] for col in cursor.description]
rows = cursor.fetchall()
df_beta = pd.DataFrame(rows, columns=columns)
filename = "beta_records_2025-05-21.csv"
df_beta.to_csv(filename, index=False)
df_beta_original = df_beta

# %%time
# Selecting all the columns from the seed and harvest records
cursor.execute("""
SELECT FARM_ID, PADDOCK_ID, RECORD_ID, CROP_TYPE, APPLICATION_DATE, YIELD_KG_PER_HA
FROM FORAGECASTER_PROD.PADDOCK_SOURCE.HARVEST_RECORD
""")
columns = [col[0] for col in cursor.description]
rows = cursor.fetchall()
df_harvest = pd.DataFrame(rows, columns=columns)
filename = "harvest_records_2025-05-21.csv"
df_harvest.to_csv(filename, index=False)
df_harvest_original = df_harvest

# %%time
# Selecting all the useful columns from the seed records
cursor.execute("""
SELECT FARM_ID, PADDOCK_ID, RECORD_ID, GRASS_TYPE, APPLICATION_DATE
FROM FORAGECASTER_PROD.PADDOCK_SOURCE.SEED_RECORD
""")
columns = [col[0] for col in cursor.description]
rows = cursor.fetchall()
df_seed = pd.DataFrame(rows, columns=columns)
filename = "seed_records_2025-05-21.csv"
df_seed.to_csv(filename, index=False)
df_seed_original = df_seed

# Making the beta columns match the seeding and harvesting column names
beta_columns_mapping = {
    "FARM_ID": "unencryped_farm_id",
    "ENCRYPTED_FARM_ID": "FARM_ID",
    "PADDOCK_ID": "unencryped_paddock_id",
    "ENCRYPTED_PADDOCK_ID": "PADDOCK_ID",
}
beta_columns_filtered = ['PADDOCK_ID', 'CREATION_DATE', 'LAST_MODIFIED_DATE', 'TITLE', 'CROP_TYPE', 'PASTURE_STATE', 'LATITUDE', 'LONGITUDE', 'WKT']
df_beta = df_beta_original.rename(columns = beta_columns_mapping)[beta_columns_filtered]

seed_columns_mapping = {
    'APPLICATION_DATE':'SOWING_DATE',
    'GRASS_TYPE':'SOWING_CROP_TYPE'
}
seed_columns_filtered = ["PADDOCK_ID", 'SOWING_CROP_TYPE',  'SOWING_DATE']
df_seed = df_seed.rename(columns = seed_columns_mapping)[seed_columns_filtered]

harvest_columns = {
    'APPLICATION_DATE':'HARVEST_DATE',
    'CROP_TYPE':'HARVEST_CROP_TYPE'
}
harvest_columns_filtered = ["PADDOCK_ID", 'HARVEST_CROP_TYPE', 'HARVEST_DATE', 'YIELD_KG_PER_HA']
df_harvest = df_harvest.rename(columns = harvest_columns)[harvest_columns_filtered]

# Convert from bytes to hex so I can merge them
df_seed['PADDOCK_ID'] = df_seed['PADDOCK_ID'].apply(lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else x)
df_harvest['PADDOCK_ID'] = df_harvest['PADDOCK_ID'].apply(lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else x)
df_beta['PADDOCK_ID'] = df_beta['PADDOCK_ID'].apply(lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else x)

# Inner join on Paddock_ID to find just the beta paddocks with sowing and harvest dates
df_beta_seed = df_beta.merge(df_seed)
df_beta_harvest = df_beta.merge(df_harvest)

# Concat the sowing and harvest dataframes
df_beta_combined = pd.concat([df_beta_seed, df_beta_harvest]).drop_duplicates()
df_beta_combined.to_csv("beta_sowing_harvest_2025-05-22.csv", index=False)

print("Total number of paddock harvest dates:", len(df_harvest.drop_duplicates()))
print("Total number of paddock yield:", len(df_harvest.drop_duplicates()['YIELD_KG_PER_HA'][df_harvest.drop_duplicates()['YIELD_KG_PER_HA'] > 0]))
print("Total number of paddock sowing dates:", len(df_seed.drop_duplicates()))
print("Number of beta paddocks:", len(df_beta.drop_duplicates()))
print("Number of beta paddock harvest dates:", len(df_beta_harvest.drop_duplicates()))
print("Number of beta paddock yields:", len(df_beta_harvest.drop_duplicates()['YIELD_KG_PER_HA'][df_beta_harvest.drop_duplicates()['YIELD_KG_PER_HA'] > 0]))
print("Number of beta paddock sowing dates:", len(df_beta_seed.drop_duplicates()))



# Save as a geopackage for viewing in QGIS
gdf = gpd.GeoDataFrame(df_beta_combined, geometry=df_beta_combined['WKT'].apply(wkt.loads), crs='EPSG:4326').drop(columns=["WKT"])
gdf['YIELD_KG_PER_HA'] = gdf['YIELD_KG_PER_HA'].astype(float)
gdf.to_file("beta_sowing_harvest_2025-05-22.gpkg")

df_harvest['YIELD_KG_PER_HA'][df_harvest['YIELD_KG_PER_HA'].notna()] > 0

len(df_beta_harvest.drop_duplicates()['YIELD_KG_PER_HA'][df_beta_harvest.drop_duplicates()['YIELD_KG_PER_HA'] > 0])



# +
# Export the beta paddock boundaries
cursor.execute("""
SELECT DISTINCT PADDOCK_ID, ST_ASWKT(GEOMETRY) AS WKT
FROM FORAGECASTER_PROD.PADDOCK_10_24_2024.BETA_PADDOCKS;
""")

rows = cursor.fetchall()

gdf = gpd.GeoDataFrame(
    [{'PADDOCK_ID': row[0], 'geometry': wkt.loads(row[1])} for row in rows],
    crs='EPSG:4326'
)

filename = "paddocks.geojson"
gdf.to_file(filename, driver="GeoJSON")
print("Saved", filename)
# -

# Create a DataFrame and convert WKT to geometry
df = pd.DataFrame(rows, columns=columns)
df_clean = df.drop(columns=['GEOMETRY', 'WKT', 'ENCRYPTED_FARM_ID', 'ENCRYPTED_PADDOCK_ID'])
gdf = gpd.GeoDataFrame(df_clean, geometry=df['WKT'].apply(wkt.loads), crs='EPSG:4326')

# Save as GeoPackage
filename = "paddocks.gpkg"
gdf.to_file(filename, driver="GPKG", layer="PADDOCK_ID")
print("Saved", filename)

# +
# Connect to the database
conn = snowflake.connector.connect(
    user='ANU_CHRISTOPHER',
    password=password,
    account='pawhkoa-mp20061',
    warehouse='FORAGECASTER_WH',
    database='FORAGECASTER_PROD',
    schema='PADDOCK_10_24_2024',
)
cursor = conn.cursor()

# Get all the colummn names in FORAGECASTER_PROD
cursor.execute("""
SELECT 
    table_schema,
    table_name,
    column_name,
    data_type
FROM 
    FORAGECASTER_PROD.INFORMATION_SCHEMA.COLUMNS
ORDER BY 
    table_schema, table_name, ordinal_position
""")
rows = cursor.fetchall()

# See if any of them are sowing or harvest records
df = pd.DataFrame(rows, columns=["Schema", "Table", "Column", "Data Type"])

# -

keywords = "sowing|harvest|seed|yield|death|cultivation"
matches = df[df["Column"].str.contains(keywords, case=False, na=False)]
print(f"Number of sowing or harvest columns: {len(matches)}")

matches

keywords = "sowing|harvest|seed|yield|death"
matches = df[df["Schema"].str.contains(keywords, case=False, na=False)]
matches

filename = "Agriwebb_Schema.csv"
df.to_csv(filename)
print("Saved", filename)

df_original = df

# +
# Simplify to just two columns
df["Table (with Schema)"] = df["Schema"] + "." + df["Table"]
df["Column (with Type)"] = df["Column"] + " (" + df["Data Type"] + ")"

# Optional: select just the new columns and reorder if you'd like
df_cleaned = df[["Table (with Schema)", "Column (with Type)"]]

# +
# Create a list of columns in each table
df["col_idx"] = df.groupby("Table (with Schema)").cumcount()

# Pivot so each field name becomes a new column
df_wide = df.pivot(index="Table (with Schema)", columns="col_idx", values="Column")

# Optionally rename columns: 0 => Column 1, 1 => Column 2, etc.
df_wide.columns = [f"Column {i+1}" for i in df_wide.columns]

# Reset index so "Table" is a normal column again
df_wide.reset_index(inplace=True)

# Export to Excel
df_wide.to_excel("snowflake_table_columns_wide.xlsx", index=False)

print("Exported to snowflake_table_columns_wide.xlsx")

# +
# Create a binary presence matrix: 1 if table has column, else 0
df_binary = (
    df[["Table (with Schema)", "Column"]]
    .assign(present=1)
    .pivot_table(index="Table (with Schema)", columns="Column", values="present", fill_value=0)
)

# Optional: order columns by how many tables contain that column
df_binary = df_binary.loc[:, df_binary.sum().sort_values(ascending=False).index]

# Reset index so "Table" becomes a column again
df_binary.reset_index(inplace=True)

# Export to Excel
df_binary.to_excel("snowflake_table_column_matrix.xlsx", index=False)

print("Exported to snowflake_presence_matrix.xlsx")


# +
# Get all the data from the beta farms
beta_columns = ('FARM_ID',
'PADDOCK_ID',
'TITLE',
'CROP_TYPE',
'ARABLELANDSIZE_HA',
'LANDSIZE_HA',
'PASTURE_STATE',
'LATITUDE',
'LONGITUDE',
'AREA_DATE',
'CREATION_DATE',
'LAST_MODIFIED_DATE')
cursor.execute(f"""
SELECT {", ".join(beta_columns)}
FROM FORAGECASTER_PROD.PADDOCK_4_28_2025.BETA_PADDOCKS;
""")
rows = cursor.fetchall()

df = pd.DataFrame(rows, columns=beta_columns)
filename = "BETA_PADDOCKS_4_28_2025.csv"
df.to_csv(filename, index=False)
print("Saved", filename)
# -

df_binary

gpd.read_file('BETA_PADDOCKS_10_24_2024.gpkg')

# !ls
