import geopandas as gpd
import fiona

# Enable KML support which is disabled by default
# Thank you Weiji14 for solving this one: https://gis.stackexchange.com/questions/114066/handling-kml-csv-with-geopandas-drivererror-unsupported-driver-ucsv
fiona.drvsupport.supported_drivers['kml'] = 'rw' 
fiona.drvsupport.supported_drivers['KML'] = 'rw' 
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
fiona.drvsupport.supported_drivers['libkml'] = 'rw' 

filename = "/Users/christopherbradley/Desktop/World_Koppen.kml"
gdf = gpd.read_file(filename)

gdf
