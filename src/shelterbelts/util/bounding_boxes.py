# +
import os
import argparse
import glob

import rasterio
import fiona
from shapely.geometry import box, mapping, Point
from pyproj import Transformer
import geopandas as gpd
from shapely.prepared import prep


# -

# Should move the api.barra_bbox function to here for consistency. 

def bounding_boxes(folder, outdir=".", stub="TEST", filetype='.tif'):
    """Download a gpkg of bounding boxes for all the tif files in a folder

    Parameters
    ----------
        folder: Folder containing lots of tifs that we want to extract the bounding box from
        outdir: The output directory to save the results 
        stub: Prefix for output file 

    Downloads
    ---------
        footprint_gpkg: A gpkg with the bounding box of each tif file and corresponding filename
        centroid_gpkg: A gpkg of the centroid of each tif file (this can be easier to view when there are lots of small tif files and you're zoomed out)
    
    """

    tif_files = glob.glob(os.path.join(filepath, f"*{filetype}*"))
    
    footprint_gpkg = f"{outdir}/{stub}_footprints.gpkg"
    centroid_gpkg = f"{outdir}/{stub}_centroids.gpkg"
    
    footprint_crs = 'EPSG:3857'
    centroid_crs = 'EPSG:4326'
    
    footprint_schema = {
        'geometry': 'Polygon',
        'properties': {'filename': 'str'}
    }
    
    centroid_schema = {
        'geometry': 'Point',
        'properties': {'filename': 'str'}
    }
    
    with fiona.open(footprint_gpkg, 'w', crs=footprint_crs, schema=footprint_schema) as fp_dst, \
         fiona.open(centroid_gpkg, 'w', crs=centroid_crs, schema=centroid_schema) as ct_dst:
    
        for i, tif in enumerate(tif_files):
            if i % 10 == 0:
                print(f"Working on tiff {i}/{len(tif_files)}")
            try:
                with rasterio.open(tif) as src:
                    bounds = src.bounds
                    src_crs = src.crs
        
                    # Transform bounds to EPSG:3857
                    footprint_transformer = Transformer.from_crs(src_crs, footprint_crs, always_xy=True)
                    minx, miny = footprint_transformer.transform(bounds.left, bounds.bottom)
                    maxx, maxy = footprint_transformer.transform(bounds.right, bounds.top)
                    geom = box(minx, miny, maxx, maxy)
        
                    # Write footprint
                    fp_dst.write({
                        'geometry': mapping(geom),
                        'properties': {'filename': os.path.basename(tif)}
                    })
        
                    # Get centroid in original CRS
                    centroid = geom.centroid
        
                    # Transform centroid to EPSG:4326
                    centroid_transformer = Transformer.from_crs(footprint_crs, centroid_crs, always_xy=True)
                    lon, lat = centroid_transformer.transform(centroid.x, centroid.y)
                    point = Point(lon, lat)
        
                    # Write centroid
                    ct_dst.write({
                        'geometry': mapping(point),
                        'properties': {'filename': os.path.basename(tif)}
                    })
            except Exception:
                    print(f"Could not open {tif}")
                    
    print(f"Saved: {footprint_gpkg}")
    print(f"Saved: {centroid_gpkg}")


# +
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


# -

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filepath', help='Folder containing lots of tifs that we want to extract the bounding box from')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', help='Prefix for output file')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arguments()

    filepath = args.filepath
    outdir = args.outdir
    stub = args.stub

    bounding_boxes(filename, outdir, stub)

# +
# filepath = "/Users/christopherbradley/Documents/PHD/Data/Worldcover_Australia"
# stub = "worldcover"
# outdir = "../../../outdir"
# bounding_boxes(filepath, outdir, stub)

# filepath = "/scratch/xe2/cb8590/Worldcover_Australia"
# stub = "Worldcover_Australia"
# outdir = "/g/data/xe2/cb8590/Outlines"
# bounding_boxes(filepath, outdir, stub)

# Footprints currently aren't working with the .asc files, but centroids are for some reason.
# filepath = '/g/data/xe2/cb8590/NSW_5m_DEMs'
# stub = 'NSW_5m_DEMs'
# outdir = "/g/data/xe2/cb8590/Outlines"
# bounding_boxes(filepath, outdir, stub, filetype='.asc')

# +
# Notes on number of tiles:
# 1 million in the original barra_bboxs
# 440k tiles in Australia
# 100k in the NSW bbox
# 50k in NSW
# -

# %%time
filepath_barra_bbox_full = '/scratch/xe2/cb8590/tmp/barra_bboxs.gpkg'
gdf = gpd.read_file(filepath_barra_bbox_full)

# %%time
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

# %%time
# Create a geopackage for each state
for state, abbreviation in state_mapping.items():
    geom = gdf2.loc[gdf2['STE_NAME21'] == 'New South Wales', 'geometry'].unary_union
    gdf_sindex = gdf.sindex
    possible_matches_index = list(gdf_sindex.intersection(geom.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    geom_prep = prep(geom)
    mask = possible_matches.geometry.map(geom_prep.intersects)
    gdf_state = possible_matches[mask]
    
    filename = os.path.join('/g/data/xe2/cb8590/Outlines/BARRA_bboxs', f'barra_bboxs_{abbreviation}.gpkg')
    gdf_nsw.to_file(filename)
    print(f"Saved: {filename}")

# %%time
filename_aus = '/g/data/xe2/cb8590/Outlines/AUS_2021_AUST_GDA2020.shp'
gdf3 = gpd.read_file(filename_aus)

geom_prep = prep(gdf3.loc[0,'geometry'])

# %%time
mask = gdf.geometry.map(geom_prep.intersects)
# Only took 1 min, I love shapely's prep function

gdf_aus = gdf[mask]
len(gdf_aus)

# %%time
filename = os.path.join('/g/data/xe2/cb8590/Outlines/BARRA_bboxs', f'barra_bboxs_aus.gpkg')
gdf_aus.to_file(filename)
print(f"Saved: {filename}")


