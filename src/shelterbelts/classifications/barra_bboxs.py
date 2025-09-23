import os
import geopandas as gpd
from shapely.prepared import prep
from shapely.geometry import box
import xarray as xr
import numpy as np


# The code below here is uncleaned, and doesn't really belong in the pipeline yet. And I'm only using the code above in the classifications pipeline, so I think it belongs there.
def create_index(gpkg, tmpdir):
    """Creates a geojson from the gpkg for my tile merging function in apis.canopy_height"""
    gdf = gpd.read_file(gpkg)
    gdf['tile'] = [filename.split('.')[0] for filename in gdf['filename']]
    gdf = gdf[['tile', 'geometry']]
    gdf = gdf.to_crs('EPSG:4326')
    filename = os.path.join(tmpdir, 'tiles_global.geojson')
    gdf.to_file(filename)
    print("Saved:", filename)
    return gdf

# create_index('/g/data/xe2/cb8590/Outlines/Worldcover_Australia_footprints.gpkg', '/scratch/xe2/cb8590/Worldcover_Australia')
# create_index('/g/data/xe2/cb8590/Outlines/global_canopy_height_footprints.gpkg', '/scratch/xe2/cb8590/Global_Canopy_Height')
# = (trying to get code to treat the above lines as commented out code instead of markdown)


# +

# # +
def pixel_bbox(i, j, transform):
    """Get the bbox of a specific pixel"""
    x0, y0 = transform * (j, i)        # top-left corner
    x1, y1 = transform * (j + 1, i + 1)  # bottom-right corner
    return box(x0, y1, x1, y0) 

def get_barra_bboxs(filename=None):
    """Create a gdf of all the bboxs in the BARRA dataset"""
    # Could swap this to the Thredds version to make it work not on NCI
    # Might want to make ds an input if I need to do this on a different dataset
    url = f"/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day/uas/latest/uas_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_202001-202001.nc"
    ds = xr.open_dataset(url, engine="netcdf4")

    # Get properties
    transform = ds.rio.transform()
    crs = ds.rio.crs
    ny, nx = ds.rio.shape

    # Create the gdf
    bboxes = []
    for i in range(ny):
        for j in range(nx):
            geom = pixel_bbox(i, j, transform)
            bboxes.append(geom)
    gdf_all = gpd.GeoDataFrame(geometry=bboxes, crs=crs)  # Took about 40 secs

    if filename:
        gdf_all.to_file(filename)  # Took about 5 mins and final output was about 200MB. Could make it smaller by cropping to the Australia border.
        print(f"Saved: {filename}")
        # Then I created smaller files for testing by selecting the relevant tiles in QGIS and Export > Save Selected Features as 

    return gdf_all

# filename = '/scratch/xe2/cb8590/tmp/barra_bboxs.gpkg'
# get_barra_bboxs(filename)

def crop_barra_bboxs():
    # This is the code I used to create barra_bboxs_aus and the state versions
    filepath_barra_bbox_full = '/scratch/xe2/cb8590/tmp/barra_bboxs.gpkg' # Takes 3 mins to load
    gdf = gpd.read_file(filepath_barra_bbox_full)

    filename_state_boundaries = '/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp'
    gdf2 = gpd.read_file(filename_state_boundaries)

    state_mapping = {"New South Wales": "nsw",
                     "Victoria":"vic",
                     "Queensland":"qld",
                     "South Australia":"sa",
                     "Western Australia":"wa",
                     "Tasmania":"tas",
                     "Northern Territory":"nt",
                     "Australian Capital Territory":"act"
    }

    # Create a geopackage for each state
    for state, abbreviation in state_mapping.items():
        geom = gdf2.loc[gdf2['STE_NAME21'] == state, 'geometry'].unary_union
        gdf_sindex = gdf.sindex
        possible_matches_index = list(gdf_sindex.intersection(geom.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        geom_prep = prep(geom)
        mask = possible_matches.geometry.map(geom_prep.intersects)
        gdf_state = possible_matches[mask]

        filename = os.path.join('/g/data/xe2/cb8590/Outlines/BARRA_bboxs', f'barra_bboxs_{abbreviation}.gpkg')
        gdf_state.to_file(filename)
        print(f"Saved: {filename}")

    # Create a geopackage of the tiles inside Australia
    filename_aus = '/g/data/xe2/cb8590/Outlines/AUS_2021_AUST_GDA2020.shp'
    gdf3 = gpd.read_file(filename_aus)
    geom_prep = prep(gdf3.loc[0,'geometry'])
    mask = gdf.geometry.map(geom_prep.intersects) # Only took 1 min, I love shapely's prep function
    gdf_aus = gdf[mask]
    filename = os.path.join('/g/data/xe2/cb8590/Outlines/BARRA_bboxs', f'barra_bboxs_aus.gpkg')
    gdf_aus.to_file(filename)
    print(f"Saved: {filename}")

    # Notes on number of tiles:
    # 1 million in the original barra_bboxs
    # 440k tiles in Australia boundary
    # 100k in the NSW bbox
    # 50k in NSW boundary


# -

def half_degree_geopackage():
    """This code created a geopackage of 0.5 x 0.5 degree tiles that I clipped in QGIS and used as input in ELVIS to request ~128GB downloads each"""
    filename_state_boundaries = '/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp'
    gdf2 = gpd.read_file(filename_state_boundaries)
    nsw_bounds = gdf2.loc[gdf2['STE_NAME21'] == 'New South Wales', 'geometry'].total_bounds
    
    # Step 1: get bbox of NSW
    nsw_geom = gdf2.loc[gdf2['STE_NAME21'] == 'New South Wales', 'geometry']
    minx, miny, maxx, maxy = nsw_geom.total_bounds
    
    # Step 2: snap to whole numbers (floor for min, ceil for max)
    minx = np.floor(minx)
    miny = np.floor(miny)
    maxx = np.ceil(maxx)
    maxy = np.ceil(maxy)
    
    # Step 3: create 1° x 1° tiles
    tiles = []
    interval = 0.5
    for x in np.arange(minx, maxx, interval):
        for y in np.arange(miny, maxy, interval):
            tiles.append(box(x, y, x+interval, y+interval))
    
    grid = gpd.GeoDataFrame(geometry=tiles, crs=gdf2.crs)
    
    # Step 4: clip to bbox (optional: only keep tiles intersecting NSW bbox)
    bbox_poly = box(minx, miny, maxx, maxy)
    grid = grid[grid.intersects(bbox_poly)]
    
    # Step 5: save to gpkg
    grid.to_file(f"nsw_tiles_half_degree.gpkg", layer="tiles", driver="GPKG")


# +
def geopackage_km(filename_state_boundaries, state='New South Wales', tile_size=30000, crs=7855, outdir='/scratch/xe2/cb8590/lidar/polygons/elvis_inputs/'):
    """Create a geopackage of bounding boxes of a given size covering a given area"""
    gdf2 = gpd.read_file(filename_state_boundaries)
    nsw = gdf2.loc[gdf2['STE_NAME21'] == state].to_crs(crs)
    
    minx, miny, maxx, maxy = nsw.total_bounds
    if state == 'New South Wales':
        minx, miny, maxx, maxy = -85000, 5845000, 1250000, 6870000  # These are nicer boundaries for NSW. Will need to unhardcode if I want to reuse for other states
    xs = np.arange(minx, maxx, tile_size)
    ys = np.arange(miny, maxy, tile_size)
    
    tiles = [box(x, y, x + tile_size, y + tile_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=tiles, crs=nsw.crs)
    
    # Keep only tiles that intersect NSW
    geom = nsw.union_all()
    geom_prep = prep(geom)
    mask = grid.geometry.map(geom_prep.intersects)
    grid_in_nsw = grid[mask]
    print("Number of tiles:", len(grid_in_nsw))
    
    outdir_geojsons = os.path.join(outdir, f"geojsons_{tile_size}")
    os.makedirs(outdir_geojsons, exist_ok=True)
    
    filenames = []
    grid_in_nsw = grid_in_nsw.reset_index(drop=True)
    for idx, row in grid_in_nsw.iterrows():
        centroid = row.geometry.centroid
        cx, cy = map(int, (centroid.x, centroid.y))  # round or cast to int for filenames
        filename = f"{outdir_geojsons}/tile{idx}_{cx}_{cy}.geojson"
        filenames.append(filename)
        gdf = gpd.GeoDataFrame([row], crs=grid_in_nsw.crs)    
        gdf.to_file(filename, driver="GeoJSON")
        if idx % 100 == 0:
            print('Saved:', filename)
    
    grid_in_nsw["filename"] = filenames
    filename = os.path.join(outdir, f'tiles_{tile_size}_{state.replace(" ", "_")}.gpkg')
    grid_in_nsw.to_file(filename, driver="GPKG")
    print('Saved:', filename)

# filename_state_boundaries = '/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp'
# geopackage_km(filename_state_boundaries)
