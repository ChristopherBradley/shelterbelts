# Full script takes 40 secs to run
import time
start_time = time.time()

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

# Directory structure
indir = '../data/'
outdir = '../outdir/'

# %%time
# Read the data GRDC sent me
filename = f'{indir}2017-2024 NVT Yield and Grain Quality - cbradley - 01.05.2025.xlsx'
dfs = pd.read_excel(filename, sheet_name=None)

df_cereals = dfs['Cereals']
df_pulses = dfs['Pulses']
df_canola = dfs['Canola']

# +
# Combine the different crops (each one is in a separate tab of the excel spreadsheet)
df_all = pd.concat([df_cereals, df_pulses, df_canola])

# Remove the 186 trials without lat lon coordinates
df_all = df_all[df_all[['Trial GPS Lat', 'Trial GPS Long']].notna().any(axis=1)]

# Merge the lat lons for easily copying into google maps
df_all['latlon'] = df_all['Trial GPS Lat'].astype(str) + ", " + df_all['Trial GPS Long'].astype(str)
# -

# Create geometry column
geometry = [Point(xy) for xy in zip(df_all['Trial GPS Long'], df_all['Trial GPS Lat'])]
gdf = gpd.GeoDataFrame(df_all, geometry=geometry, crs='EPSG:4326')

# Remove the plot information so we just have the trial information
gdf_trials = gdf[['Year', 'Crop.Name', 'State', 'RegionName', 'SiteDescription', 'TrialCode', 'Orientation', 'Trial GPS Lat', 'Trial GPS Long', 'SowingDate',	'HarvestDate', 'Yield Mean', 'CV', 'Standard Error Difference', 'cluster', 'geometry']].drop_duplicates()
gdf_trials.to_file(f"{outdir}TrialCodes.kml", driver="KML") # For viewing in Google Earth
gdf_trials.to_file(f"{outdir}TrialCodes.gpkg") # For viewing in QGIS

# Display the crop names, years, and regions
gdf_trials['Crop.Name'].value_counts()

gdf_trials['Year'].value_counts()

gdf_trials['RegionName'].unique()


# Find paddocks where a single crop type is trialled within the paddock
def unique_crop_paddock(df_all, radius=100, grouping_method=['cluster', 'year']):
    """Remove rows where there are multiple different crop types for the same year within the given distance (meters)
    
    Parameters
    ----------
    df_all - A dataframe with at least the columns 'Trial GPS Lat', 'Trial GPS Long', 'Crop.Name', 'SowingDate'
    radius - Allowable distance in metres to have different crop names
    grouping_method - Either 'cluster', ['cluster', 'year'] or depending on whether it's ok to have crop rotations or not.

    Returns
    -------
    An equivalent dataframe after removing any rows which have different crop names within the specified distance
    
    """
    
    # Convert lat/lon to radians for haversine
    coords = np.radians(df_all[['Trial GPS Lat', 'Trial GPS Long']].values)
    
    # Convert distance in meters to radians
    eps = radius / 6371000  # Earth's radius in meters
    
    # Spatial clustering
    db = DBSCAN(eps=eps, min_samples=1, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(coords)
    df_all = df_all.copy()
    df_all['cluster'] = labels

    # Extract year
    df_all['year'] = pd.to_datetime(df_all['SowingDate']).dt.year

    # Filter clusters with a single crop per year
    valid_groups = (
        df_all.groupby(grouping_method)['Crop.Name']
        .nunique()
        .loc[lambda x: x == 1]
        .index
    )

    return df_all[df_all.set_index(grouping_method).index.isin(valid_groups)].copy()



# +
# If there are multiple points with the same crop type in the same year, then just keep one of those rows

# +
print(f"Original number of trials: {len(gdf_trials)}")

# Some examples of the amount of trials that get filtered out when using different distance threshold
gdf_100m = unique_crop_paddock(gdf_trials, 100, ['cluster', 'year'])
gdf_1000m = unique_crop_paddock(gdf_trials, 1000, ['cluster', 'year'])
gdf_100m_allyears = unique_crop_paddock(gdf_trials, 100, 'cluster')
gdf_1000m_allyears = unique_crop_paddock(gdf_trials, 1000, 'cluster')

print(f"gdf_100m: {len(gdf_100m)}")
print(f"gdf_1000m: {len(gdf_1000m)}")
print(f"gdf_100m_allyears: {len(gdf_100m_allyears)}")
print(f"gdf_1000m_allyears: {len(gdf_1000m_allyears)}")
# -

# Canola filtering
print(f"gdf_100m: {len(gdf_100m[gdf_100m['Crop.Name'] == 'Canola'])}")
print(f"gdf_1000m: {len(gdf_1000m[gdf_1000m['Crop.Name'] == 'Canola'])}")
print(f"gdf_100m_allyears: {len(gdf_100m_allyears[gdf_100m_allyears['Crop.Name'] == 'Canola'])}")
print(f"gdf_1000m_allyears: {len(gdf_1000m_allyears[gdf_1000m_allyears['Crop.Name'] == 'Canola'])}")

# Wheat filtering
print(f"gdf_100m: {len(gdf_100m[gdf_100m['Crop.Name'] == 'Wheat'])}")
print(f"gdf_1000m: {len(gdf_1000m[gdf_1000m['Crop.Name'] == 'Wheat'])}")
print(f"gdf_100m_allyears: {len(gdf_100m_allyears[gdf_100m_allyears['Crop.Name'] == 'Wheat'])}")
print(f"gdf_1000m_allyears: {len(gdf_1000m_allyears[gdf_1000m_allyears['Crop.Name'] == 'Wheat'])}")

# Huge threhold of 10km
gdf_10km = unique_crop_paddock(gdf_trials, 10000, 'cluster')
print(len(gdf_10km[gdf_10km['Crop.Name'] == 'Canola']))
print(len(gdf_10km[gdf_10km['Crop.Name'] == 'Wheat']))

# Browsing the results in QGIS and google earth
gdf_10km = gdf_10km[['Year', 'Crop.Name', 'State', 'RegionName', 'SiteDescription', 'TrialCode', 'Orientation', 'Trial GPS Lat', 'Trial GPS Long', 'SowingDate',	'HarvestDate', 'Yield Mean', 'CV', 'Standard Error Difference', 'cluster', 'geometry']].drop_duplicates()
gdf_10km_canola = gdf_10km[gdf_10km['Crop.Name'] == 'Canola']
gdf_10km_canola.to_file(f"{outdir}TrialCodes_10km_Canola.kml", driver="KML")

gdf_10km_canola


