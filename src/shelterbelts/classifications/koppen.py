import geopandas as gpd
import fiona
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import box


# Download from here: https://people.eng.unimelb.edu.au/mpeel/koppen.html
filename = "/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/Koppen/World_Koppen.kml"

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

# Quick method for saving a geometry to file
gpd.GeoDataFrame(
    {'geometry': [example_polygon]},
    crs=gdf.crs
).to_file('/Users/christopherbradley/Desktop/example.gpkg')

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

filename_koppen = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/Koppen_Australia.gpkg"
unique = joined[['geometry', 'Name_right']].drop_duplicates()
koppen_australia = unique[['Name_right', 'geometry']].rename(columns={"Name_right": "Name"})
koppen_australia.to_file(filename_koppen)

# Load that koppen data
filename_koppen = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/Koppen_Australia.gpkg"
gdf_koppen = gpd.read_file(filename_koppen)

# Load Nick's tiff centroids
filename_centroids = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/tiff_centroids_years.gpkg"
gdf_centroids = gpd.read_file(filename_centroids)

# Label each of the tif centroids with a koppen class
gdf_koppen['area'] = gdf_koppen.geometry.area
joined = gpd.sjoin(gdf_centroids, gdf_koppen, how="left", predicate="within")
joined = joined.sort_values('area', ascending=False).drop_duplicates(subset='filename')
gdf = joined[['filename','year','geometry','Name']]

# +
# Manually edit some labels to make the groups more spatially consistent
lon = gdf.geometry.x
lat = gdf.geometry.y

gdf.loc[(lon < 144) & (lat < -31), 'Name'] = 'BSk'
gdf.loc[(lon < 131), 'Name'] = 'KUN'
gdf.loc[(lat < -24) & (lat > -29) & (lon < 143.6), 'Name'] = 'BWh'
gdf.loc[(gdf['Name'] == 'CFa') & (lat > -32.9), 'Name'] = 'CFx'
gdf.loc[(gdf['Name'] == 'Cfb') & (lat > -30), 'Name'] = 'CFx'
gdf.loc[(lon > 152.5), 'Name'] = 'CFx'
names = ['CFa', 'Cfb', 'CFx', 'BWh', 'KUN', 'BSk', 'BSh']
gdf.loc[~gdf['Name'].isin(names), 'Name'] = 'Aw'
# -

full_names = {
    'CFa':'Warm oceanic NSW', 
    'Cfb':'Temperate oceanic NSW', 
    'CFx':'Warm oceanic QLD', 
    'BWh':'Warm desert QLD', 
    'KUN':'Tropical savanna NT', 
    'BSk':'Cold semi-arid NSW', 
    'BSh':'Warm semi-arid QLD',
    'Aw':'Tropical savanna QLD'
}
gdf['Full Name'] = gdf['Name'].replace(full_names)

gdf.to_file("/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/centroids_named.gpkg")





def creating_Koppen_Australia_Cleaned():
    # Remove inner polygons and just use the outer classification
    gdf = gpd.read_file('/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/Koppen/Koppen_Australia.gpkg')  # This is 'polygons' from above
    gdf['geometry'] = gdf['geometry'].buffer(0)
    def get_exterior(geom):
        if geom.geom_type == 'Polygon':
            return Polygon(geom.exterior)
        elif geom.geom_type == 'MultiPolygon':
            return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
        else:
            return geom
    
    gdf['geometry'] = gdf['geometry'].apply(get_exterior)
    to_drop = set()
    for i, geom_i in gdf.geometry.items():
        for j, geom_j in gdf.geometry.items():
            if i != j and geom_i.within(geom_j):
                to_drop.add(i)
                break
    
    gdf_clean = gdf.drop(index=list(to_drop)).reset_index(drop=True)
    
    # Step 4: Merge geometries and clean up (removes any residual holes)
    gdf_clean['geometry'] = gdf_clean['geometry'].apply(
        lambda g: g.buffer(0) if g.geom_type == 'Polygon'
        else unary_union([poly.buffer(0) for poly in g.geoms])
    )
    
    # Split the CFa polygon and merge the bottom left into the Cfb polygon
    gdf = gdf_clean.copy()
    gdf['fid'] = gdf.index
    
    # Split the CFa and reassign half of it to Cfb
    fid_CFa = 130
    fid_Cfb = 143
    gdf_cfa = gdf[gdf['fid'] == fid_CFa].copy()  # CFa
    gdf_cfb = gdf[gdf['fid'] == fid_Cfb].copy()  # Cfb
    minx, miny, maxx, maxy = gdf.total_bounds
    south_mask = box(minx, miny, 149.5, -32.9)
    cfa_geom = unary_union(gdf_cfa.geometry)
    cfa_north = cfa_geom.difference(south_mask)
    cfa_south = cfa_geom.intersection(south_mask)
    gdf.loc[gdf['fid'] == fid_CFa, 'geometry'] = cfa_north
    cfb_geom = unary_union(gdf_cfb.geometry)
    merged_cfb = unary_union([cfb_geom, cfa_south])
    gdf.loc[gdf['fid'] == fid_Cfb, 'geometry'] = merged_cfb
    gdf['geometry'] = gdf['geometry'].buffer(0)
    
    # Reassigning little names at the top
    gdf.loc[gdf['Name'].isin(['Cfa', 'Cwa', 'Af', 'Am']), 'Name'] = 'Aw'
    gdf.loc[gdf['fid'].isin([113]), 'Name'] = 'Aw'
    
    # South WA and southwest Vic becomes Southeast NSW & Tas
    gdf.loc[gdf['Name'].isin(['Csb']), 'Name'] = 'Cfb'
    
    # Mid-south categories becoming west NSW
    gdf.loc[gdf['Name'].isin(['BWk', 'Csa', None]), 'Name'] = 'BSk'
    
    # Little bits in South Autralia
    gdf.loc[gdf['fid'].isin([76, 72, 128, 75, 73, 65, 66, 67, 69, 70, 71]), 'Name'] = 'BSk'
    
    # # Single tropical pixels on east coast 
    gdf.loc[gdf['fid'].isin([129]), 'Name'] = 'Cfb'
    gdf.loc[gdf['fid'].isin([136, 131, 132, 126, 125, 77, 79]), 'Name'] = 'CFa'
    
    # South coast
    gdf.loc[gdf['fid'].isin([97, 98, 99]), 'Name'] = 'BSk'
    
    # West coast
    gdf.loc[gdf['fid'].isin([56]), 'Name'] = 'BWh'
    
    # # # Split the BWh and reassign the bottom section to BSk
    fid_CFa = 48
    fid_Cfb = 87
    gdf_cfa = gdf[gdf['fid'] == fid_CFa].copy() 
    gdf_cfb = gdf[gdf['fid'] == fid_Cfb].copy() 
    minx, miny, maxx, maxy = gdf.total_bounds
    south_mask = box(minx, miny, maxx, -32.6)
    cfa_geom = unary_union(gdf_cfa.geometry)
    cfa_north = cfa_geom.difference(south_mask)
    cfa_south = cfa_geom.intersection(south_mask)
    gdf.loc[gdf['fid'] == fid_CFa, 'geometry'] = cfa_north
    cfb_geom = unary_union(gdf_cfb.geometry)
    merged_cfb = unary_union([cfb_geom, cfa_south])
    gdf.loc[gdf['fid'] == fid_Cfb, 'geometry'] = merged_cfb
    gdf['geometry'] = gdf['geometry'].buffer(0)
    
    gdf = gdf.dissolve(by='Name', as_index=False)
    filename = '/Users/christopherbradley/Desktop/Koppen_Australia_cleaned2.gpkg'
    gdf.to_file(filename, driver='GPKG')
    print('Saved:', filename)




