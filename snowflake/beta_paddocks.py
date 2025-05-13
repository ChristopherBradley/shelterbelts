# +
# Download the paddock boundaries for all the beta farms

# +
# # !pip install snowflake-connector-python openpyxl
# -

import snowflake.connector
import geopandas as gpd
from shapely import wkt
import pandas as pd

password = ''

conn = snowflake.connector.connect(
    user='ANU_CHRISTOPHER',
    password=password,
    account='pawhkoa-mp20061',
    warehouse='FORAGECASTER_WH',
    database='FORAGECASTER_PROD',
    schema='PADDOCK_10_24_2024',
)
cursor = conn.cursor()


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

# Selecting all the columns from beta_paddocks
cursor.execute("""
SELECT *, ST_ASWKT(GEOMETRY) AS WKT
FROM FORAGECASTER_PROD.PADDOCK_10_24_2024.BETA_PADDOCKS
""")
columns = [col[0] for col in cursor.description]
rows = cursor.fetchall()

# Create a DataFrame and convert WKT to geometry
df = pd.DataFrame(rows, columns=columns)
df_clean = df.drop(columns=['GEOMETRY', 'WKT', 'ENCRYPTED_FARM_ID', 'ENCRYPTED_PADDOCK_ID'])
gdf = gpd.GeoDataFrame(df_clean, geometry=df['WKT'].apply(wkt.loads), crs='EPSG:4326')

gdf.dtypes

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
keywords = "sowing|harvest|seed|yield|death"
matches = df[df["Column"].str.contains(keywords, case=False, na=False)]
print(f"Number of sowing or harvest columns: {len(matches)}")
# -

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
