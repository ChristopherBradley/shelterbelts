# +
# Downloading canopy height for all of Australia
# I downloaded all of Worldcover manually from https://viewer.esa-worldcover.org/worldcover (can only do 100 tiles at a time, so it took 2 downloads)

import geopandas as gpd


# -

def filter_canopy_height():
    """Used this code to generate the tiles_aus.gpkg"""
    # Filepaths

    # Downloaded from here: https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/tiles.geojson
    canopy_height_tiles = '/Users/christopherbradley/repos/PHD/shelterbelts/tiles_global.geojson' 
    
    # Downloaded from here: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
    aus_boundary = '/Users/christopherbradley/Documents/PHD/Data/AUS_2021_AUST_SHP_GDA2020/AUS_2021_AUST_GDA2020.shp'
    
    outpath = '/Users/christopherbradley/Documents/PHD/Data/Global_Canopy_Height/tiles_aus.gpkg'

    # Load the canopy height and australia boundary
    df_canopy_height = gpd.read_file(canopy_height_tiles)
    df_aus_boundary = gpd.read_file(aus_boundary)
    df_aus_boundary = df_aus_boundary.to_crs(df_canopy_height.crs)

    # Filter the canopy height tiles and save a geopackage
    aus_geom = df_aus_boundary.iloc[0].geometry
    df_canopy_height_filtered = df_canopy_height[df_canopy_height.intersects(aus_geom)]
    df_canopy_height_filtered.to_file(outpath)

# Setup a script that can be run from the command line

# import and run the canopy height download into scratch for all of the Australia
