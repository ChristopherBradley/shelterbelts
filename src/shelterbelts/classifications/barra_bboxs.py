# +
import os
import shutil
from pathlib import Path
import geopandas as gpd
from shapely.prepared import prep
from shapely.geometry import box, Polygon

import xarray as xr
import numpy as np


<<<<<<< HEAD
# -

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


=======
>>>>>>> 93f0f9f34ff7b596c235cd4382ebf0fd65855e7d
# +

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

def sub_gpkgs():
    """Create smaller gpkgs for passing to multiple prediction jobs at once"""
    # Input / output paths
    input_file = "/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_nsw.gpkg"
    output_dir = "/g/data/xe2/cb8590/Outlines/BARRA_bboxs/BARRA_bboxs_nsw"
    os.makedirs(output_dir, exist_ok=False)
    
    # Load the GeoDataFrame
    gdf = gpd.read_file(input_file)
    gdf['stub'] = [f"{geom.centroid.y:.2f}-{geom.centroid.x:.2f}".replace(".", "_")[1:] for geom in gdf['geometry']]

    # Remove tiles that have already been processed
    proc_dir = Path("/scratch/xe2/cb8590/barra_trees_2020")
    processed_stubs = {
        f.stem.replace("_predicted", "") for f in proc_dir.glob("*_predicted.tif")
    }
    gdf = gdf[~gdf["stub"].isin(processed_stubs)]
        
    # Chunk size
    chunk_size = 500
    
    # Split into chunks and save
    for start in range(0, len(gdf), chunk_size):
        end = min(start + chunk_size, len(gdf))
        chunk = gdf.iloc[start:end]
        out_file = os.path.join(output_dir, f"BARRA_bboxs_nsw_{start}-{end}.gpkg")
        chunk.to_file(out_file, driver="GPKG")
        print(f"Saved {out_file}")


# +
def geopackage_km(filename_state_boundaries, state='New South Wales', tile_size=30000, crs=7855, outdir='/scratch/xe2/cb8590/lidar/polygons/elvis_inputs/'):
    """Create a geopackage of bounding boxes of a given size covering a given area"""
    gdf2 = gpd.read_file(filename_state_boundaries)
    nsw = gdf2.loc[gdf2['STE_NAME21'] == state].to_crs(crs)
    
    minx, miny, maxx, maxy = nsw.total_bounds
    if state == 'New South Wales':
        minx, miny, maxx, maxy = -85000, 5845000, 1250000, 6870000  # These are nicer boundaries for NSW. Will need to unhardcode if I want to reuse for other states. 
        # minx, miny, maxx, maxy = -84000, 5844000, 1250000, 6870000  # Exactly matching up with the 2kmx2km NSW grid

    xs = np.arange(minx, maxx, tile_size)
    ys = np.arange(miny, maxy, tile_size)
    
    tiles = [box(x, y, x + tile_size, y + tile_size) for y in ys for x in xs]
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
        filenames.append(f'tile{idx}_{cx}_{cy}')
        gdf = gpd.GeoDataFrame([row], crs=grid_in_nsw.crs).to_crs(4326)
        gdf.to_file(filename, driver="GeoJSON")
        if idx % 100 == 0:
            print('Saved:', filename)
    
    grid_in_nsw["filename"] = filenames
    filename = os.path.join(outdir, f'tiles_{tile_size}_{state.replace(" ", "_")}.gpkg')
    grid_in_nsw.to_file(filename, driver="GPKG")
    print('Saved:', filename)

# filename_state_boundaries = '/g/data/xe2/cb8590/Outlines/STE_2021_AUST_GDA2020.shp'
# geopackage_km(filename_state_boundaries, tile_size=30000)
# -
def single_boundary():
    """Creating a single boundary for NSW"""
    gdf = gpd.read_file('/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/Australia State Boundaries/STE_2021_AUST_GDA2020.shp')
    multipolygon = gdf.iloc[0]['geometry']
    largest_polygon = max(multipolygon.geoms, key=lambda p: p.area)
    polygon_no_holes = Polygon(largest_polygon.exterior)
    gdf_no_holes = gpd.GeoDataFrame(
        [gdf.iloc[0].to_dict()],  # Copy all the attributes from the original row
        geometry=[polygon_no_holes],
        crs=gdf.crs  # Preserve the coordinate reference system
    )
    gdf_no_holes.to_file('/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/NSW_Border_Largest_Polygon.shp')


