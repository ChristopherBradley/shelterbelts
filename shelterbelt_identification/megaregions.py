import geopandas as gpd
import fiona
import pandas as pd

# Download from here: https://people.eng.unimelb.edu.au/mpeel/koppen.html
filename = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/World_Koppen.kml"

# Enable KML support which is disabled by default
# https://gis.stackexchange.com/questions/114066/handling-kml-csv-with-geopandas-drivererror-unsupported-driver-ucsv
fiona.drvsupport.supported_drivers['kml'] = 'rw' 
fiona.drvsupport.supported_drivers['KML'] = 'rw' 
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
fiona.drvsupport.supported_drivers['libkml'] = 'rw' 

# +
# %%time
# Need to read each layer in the kml file (a single read will miss a lot of the layers)
# https://gis.stackexchange.com/questions/446036/reading-kml-with-geopandas
gdf_list = []
for layer in fiona.listlayers(filename):    
    gdf = gpd.read_file(filename, driver='LIBKML', layer=layer)
    gdf_list.append(gdf)

gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
# -

gdf.to_file('../data/World_Koppen.gpkg', driver='GPKG')


# Separate points and polygons
points = gdf[gdf.geometry.type == 'Point']
polygons = gdf[gdf.geometry.type == 'Polygon']

# Spatial join: find which point is inside which polygon
joined = gpd.sjoin(polygons, points, how='left', predicate='contains')
joined.groupby('index_right').size().value_counts()
# There are 368 polygons with 2 points, and 3 polygons with 3 points. This might not matter after we filter to Australia though

# Australia boundary downloaded from here:
# https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
filename_aus = "/Users/christopherbradley/Documents/PHD/Data/AUS_2021_AUST_SHP_GDA2020/AUS_2021_AUST_GDA2020.shp" 
australia = gpd.read_file(filename_aus)

australia_gdf = australia.to_crs(polygons.crs)
australia_geom = australia_gdf.geometry.iloc[0]

# %%time
polygons = polygons[polygons.intersects(australia_geom)]
# Took 46 secs to clip the polygons to Australia

joined = gpd.sjoin(polygons, points, how='left', predicate='contains')
joined.groupby('index_right').size().value_counts()
# There are now just 10 polygons with 2 points. 

# Fix up the rows where names and polygons were overlapping incorrectly (verified by looking at the kml file in google earth)
joined.loc[joined['Name_left'] == '1452', 'Name_right'] = 'BSh'
joined.loc[joined['Name_left'] == '1138', 'Name_right'] = 'BWh'
joined.loc[joined['Name_left'] == '1622', 'Name_right'] = 'BSk'
joined.loc[joined['Name_left'] == '2229', 'Name_right'] = 'CFa'

# Double check the lengths are the same
print((len(joined[['geometry']].drop_duplicates())))
print(len(joined[['geometry', 'Name_right']].drop_duplicates()))

filename = "~/Desktop/Koppen_Australia.gpkg"
koppen_australia = joined[['Name_right', 'geometry']].rename(columns={"Name_right": "Name"})
koppen_australia.to_file(filename)


