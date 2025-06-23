# +
# Download the paddock boundaries for all the beta farms

# +
# # !pip install snowflake-connector-python openpyxl

# +
# Can browse the snowflake database here: 
# https://ft76388.ap-southeast-2.snowflakecomputing.com/oauth/authorize?response_type=code&client_id=KEaAv5tJAYc1%2BBqySu5DRtAL1DQ%2Frw%3D%3D&scope=refresh_token&state=%7B%22isSecondaryUser%22%3Afalse%2C%22csrf%22%3A%224b3cccb4%22%2C%22url%22%3A%22https%3A%2F%2Fft76388.ap-southeast-2.snowflakecomputing.com%22%2C%22windowId%22%3A%220024ce81-ff51-4550-807d-363c668eb3db%22%2C%22classicUIUrl%22%3A%22https%3A%2F%2Fft76388.ap-southeast-2.snowflakecomputing.com%22%2C%22browserUrl%22%3A%22https%3A%2F%2Fapp.snowflake.com%2Fpawhkoa%2Fmp20061%22%2C%22originator%22%3A%22started-by-cb100-2025-04-18T00%3A16%3A47.349158261Z%22%2C%22oauthNonce%22%3A%22eXbXdrx95bRq6Eod%22%7D&redirect_uri=https%3A%2F%2Fapps-api.c1.ap-southeast-2.aws.app.snowflake.com%2Fcomplete-oauth%2Fsnowflake&code_challenge=vnflT2KKhidHSfSgmW-ukp03c1xPJ_fG5wCBdyEeNCk&code_challenge_method=S256
# -

import time
start_time = time.time()

import snowflake.connector
import geopandas as gpd
from shapely import wkt
import pandas as pd

# +
# Credentials
from credentials import password

# Directory structure
indir = '../data/'
outdir = '../outdir/'
# -

# Trying a new method. Doesn't work because it redirects to Single Sign On, when I need the snowflake instead. 
conn = snowflake.connector.connect(
    user='ANU_CHRISTOPHER',
    password=password,
    authenticator='externalbrowser',
    account='pawhkoa-mp20061',
    warehouse='FORAGECASTER_WH',
    database='FORAGECASTER_PROD',
    schema='PADDOCK_10_24_2024',
)

# This old method used to work, but doesn't work now that MFA is enforced as of 23 June 2025
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
df_beta_combined.to_csv("beta_sowing_harvest.csv", index=False)

# Some stats about the data we have
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
gdf.to_file(f"{outdir}beta_sowing_harvest.gpkg")

# Export the beta paddock boundaries
cursor.execute("""
SELECT DISTINCT PADDOCK_ID, ST_ASWKT(GEOMETRY) AS WKT
FROM FORAGECASTER_PROD.PADDOCK_10_24_2024.BETA_PADDOCKS;
""")
rows = cursor.fetchall()

# Create a DataFrame and convert WKT to geometry
df = pd.DataFrame(rows, columns=columns)
df_clean = df.drop(columns=['GEOMETRY', 'WKT', 'ENCRYPTED_FARM_ID', 'ENCRYPTED_PADDOCK_ID'])
gdf = gpd.GeoDataFrame(df_clean, geometry=df['WKT'].apply(wkt.loads), crs='EPSG:4326')

# Save as GeoPackage
filename = f"{outdir}paddocks.gpkg"
gdf.to_file(filename, driver="GPKG", layer="PADDOCK_ID")
print("Saved", filename)

end_time = time.time()
end_time - start_time
