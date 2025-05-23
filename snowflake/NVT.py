import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import numpy as np
import geopandas as gpd

# Enable KML support which is disabled by default
# https://gis.stackexchange.com/questions/114066/handling-kml-csv-with-geopandas-drivererror-unsupported-driver-ucsv
import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw' 
fiona.drvsupport.supported_drivers['KML'] = 'rw' 
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
fiona.drvsupport.supported_drivers['libkml'] = 'rw' 

# Make the panda displays more informative
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# %%time
filename = '/Users/christopherbradley/Documents/PHD/Data/GRDC/2017-2024 NVT Yield and Grain Quality - cbradley - 01.05.2025.xlsx'
dfs = pd.read_excel(filename, sheet_name=None)

df_cereals = dfs['Cereals']
df_pulses = dfs['Pulses']
df_canola = dfs['Canola']

# +
df_all = pd.concat([df_cereals, df_pulses, df_canola])

# Remove the 186 trials without lat lon coordinates
df_all = df_all[df_all[['Trial GPS Lat', 'Trial GPS Long']].notna().any(axis=1)]

df_all['latlon'] = df_all['Trial GPS Lat'].astype(str) + ", " + df_all['Trial GPS Long'].astype(str)

# +
# Spatial clustering of points directly or indirectly within 100m
coords = np.radians(df_all[['Trial GPS Lat', 'Trial GPS Long']].values)
db = DBSCAN(eps=(1/6371) / 10, min_samples=1, algorithm='ball_tree', metric='haversine')

labels = db.fit_predict(coords)
df_all['cluster'] = labels
clusters = [group for _, group in df_all.groupby('cluster')]

cluster_lengths = [len(cluster) for cluster in clusters]
largest_cluster = clusters[np.argmax(cluster_lengths)]
# -

# Create geometry column
geometry = [Point(xy) for xy in zip(df_valid['Trial GPS Long'], df_valid['Trial GPS Lat'])]
gdf = gpd.GeoDataFrame(df_valid, geometry=geometry, crs='EPSG:4326')
gdf.to_file('trials.gpkg', layer='trials', driver='GPKG')

# Remove the plot information so we just have the trial information
gdf_trials = gdf[['Year', 'Crop.Name', 'State', 'RegionName', 'SiteDescription', 'TrialCode', 'Orientation', 'Trial GPS Lat', 'Trial GPS Long', 'SowingDate',	'HarvestDate', 'Yield Mean', 'CV', 'Standard Error Difference', 'cluster', 'geometry']].drop_duplicates()
gdf_trials.to_file("TrialCodes.kml", driver="KML") # For viewing in Google Earth
gdf_trials.to_file("TrialCodes.gpkg") # For viewing in QGIS

gdf_trials['Crop.Name'].value_counts()

gdf_trials['Year'].value_counts()

gdf_trials['RegionName'].unique()
