# +
# Create a geojson file showing the outline of each training sample in QGIS
# -

import pandas as pd
import os
import re
import glob
import rasterio
from shapely.geometry import box, mapping, Point
import fiona
from pyproj import Transformer
import geopandas as gpd

pd.set_option('display.max_columns', 100)


# %%time
# Extract the bounding box and centroid for each tiff file
filepath = "/Users/christopherbradley/Documents/PHD/Data/Nick_Aus_treecover_10m"
tif_files = glob.glob(os.path.join(filepath, "*.tiff"))

footprint_geojson = "../data/tiff_footprints.geojson"
centroid_geojson = "../data/tiff_centroids.geojson"

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

with fiona.open(footprint_geojson, 'w', driver='GeoJSON', crs=footprint_crs, schema=footprint_schema) as fp_dst, \
     fiona.open(centroid_geojson, 'w', driver='GeoJSON', crs=centroid_crs, schema=centroid_schema) as ct_dst:

    for i, tif in enumerate(tif_files):
        if i % 1000 == 0:
            print(f"Working on tiff {i}/{len(tif_files)}")
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

print("GeoJSONs created:")
print(f"- {footprint_geojson}")
print(f"- {centroid_geojson}")


# Next I did the intersection of LIDAR polygons & tiff polygons in QGIS to create lidar_clipped.geojson (this was too slow in python)

# Convert from geojson to gpkg for more efficient file storage
fp = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/lidar_clipped.geojson"
out_fp = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/lidar_clipped.gpkg"

gdf = gpd.read_file(fp)
gdf = gdf.rename(columns={'fid': 'fid_original', "id":'id_original'})
gdf["fid"] = range(1, len(gdf) + 1)  # Need a unique ID column for gpkg

# Save to GeoPackage
gdf.to_file(out_fp, layer='lidar_clipped', driver="GPKG")


# %%time
fp = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/lidar_clipped.gpkg"
gdf = gpd.read_file(fp)

# Many to one relationship because each tree/no tree tiff might be covered by multiple LIDAR tiffs. But the dates should be the same. If multiple dates, then maybe I can take the most recent?
gdf[gdf['filename'] == 'g2_23456_binary_tree_cover_10m.tiff'].shape

# There are 238 rows without any name, and hence no details on the year when the LIDAR was taken
gdf[gdf['object_name_ahd'].isna() & gdf['object_name_ort'].isna() & gdf['object_name_las'].isna() & gdf['object_name'].isna()].shape

# Double check there aren't any rows with multiple names
name_cols = ['object_name_ahd', 'object_name_ort', 'object_name_las', 'object_name']
gdf[gdf[name_cols].notna().sum(axis=1) > 1].shape

# Combine the names into a single column
gdf['name'] = gdf[name_cols].bfill(axis=1).iloc[:, 0]

# Save a csv of the first 1000 names for seeing what metadata we have
gdf[:1000].to_csv("../data/gdf_1000.csv")


# +
# I'm assuming that the year is the first 4 non letter digits in the string
def extract_year(name):
    if not isinstance(name, str):
        return None
    # Some special cases
    if name.startswith('Laura22021'): 
        return '2021'
    if name.startswith('Herbert1Lidar2020') or name.startswith('Herbert2Lidar2020'):
        return '2020'
    first_number = re.search(r'\d', name).start()
    year = name[first_number:first_number+4]
    if not year[:2] == '20':
        print(name)
    assert year[:2] == '20'
    return year

# Example names: 
name1 = 'ACT2015_4ppm-C3-AHD_6626058_55_0002_0002.zip'
name2 = 'Yarrangobilly201803-LID2-C3-AHD_6106040_55_0002_0002.zip'
name3 = 'BelyandoCrossing_2013_Loc_SW_485000_7611000_1K_Las.zip'
name4 = 'Lower_Balonne_2018_Prj_SW_623000_6856000_1K_Las.zip'
name5 = None
name6 = 'Laura22021-C3-AHD_2118270_55_0001_0001.laz'
name7 = 'Herbert1Lidar2020-C3-AHD_3227992_55_0001_0001.laz'
example_names = [name1, name2, name3, name4, name5, name6]
[extract_year(name) for name in example_names]
# -

# Extract the year for each filename
gdf['year'] = gdf['name'].apply(extract_year)

# %%time
# Getting all the dates for visual inspection in excel
gdf_combined = (
    gdf.groupby('filename')
       .agg({
           'year': lambda x: ', '.join(sorted(set(filter(None, x)))),
           'name': lambda x: ', '.join(sorted(set(filter(None, x))))
       })
)
gdf_combined.to_csv("../data/gdf_filename_years.csv")

# Getting all the dates for visual inspection in excel
gdf_maxyear = (
    gdf.groupby('filename')
       .agg({
           'year': lambda x: max(int(y) for y in x.dropna())
       })
)
gdf_maxyear

gdf_maxyear.to_csv("../data/gdf_filename_maxyear.csv")

# Plotting all the LIDAR measurements
gdf['year'].dropna().astype(int).hist(bins=13)

# Plotting just the most recent lidar acquisition
gdf_maxyear['year'].hist(bins=13)

gdf_maxyear['year'].value_counts()

# Creating smaller csvs for testing the parallel processing bash script before running on everything
csv_filename = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
df_years = pd.read_csv(csv_filename, index_col='filename')
df_years_2017_2022 = df_years[df_years['year'] >= 2017]

filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_01001.csv'
df_years_2017_2022.iloc[0:1].to_csv(filename)

filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_x5.csv'
df_years_2017_2022.iloc[0:5].to_csv(filename)

filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_x100.csv'
df_years_2017_2022.sample(n=100, random_state=0).to_csv(filename)

filename ='/g/data/xe2/cb8590/Nick_outlines/gdf_2017_2022.csv'
df_years_2017_2022.to_csv(filename)


# %%time
# Should move this code into Nick_polygons.py
def extract_bbox_year():
    """I used this code to extract the bbox and year for each of Nick's tiles"""
    # Read in the bbox and year for each of the tif files
    df = pd.read_csv(filename_gdf_maxyear, index_col='filename')

    # Load the crs and bounds for each tif file
    rows = list(df[['year']].itertuples(name=None))
    rows2 = []
    for row in rows:
        tif, year = row
        filename = os.path.join(indir, tif)
        with rasterio.open(filename) as src:
            bounds = src.bounds
            crs = src.crs.to_string()
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        rows2.append((tif, year, bbox, crs))

    df = pd.DataFrame(rows2, columns=["tif", "year", "bbox", "crs"])
    df.to_csv(filename_bbox_year, index=False)
    print("Saved", filename_bbox_year)

    # Took 3 mins

# Create rows for each of the 7k bbox's for tiffs Nick provided after 2017
def prep_rows_Nick():   
    indir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
    outdir = '/scratch/xe2/cb8590/Nick_sentinel'
    filename_bbox_year = "/g/data/xe2/cb8590/Nick_outlines/nick_bbox_year_crs.csv"
    filename_gdf_maxyear = '/g/data/xe2/cb8590/Nick_outlines/gdf_filename_maxyear.csv'
    outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"

    # Load the tiff bboxs
    df = pd.read_csv(filename_bbox_year)
    df_2017_2022 = df[df['year'] >= 2017]
    print("Number of tiles between 2017-2022", len(df_2017_2022))

    # Find the tiles we have already downloaded
    sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
    sentinel_tiles = glob.glob(f'{sentinel_dir}/*')
    print("Number of sentinel tiles downloaded:", len(sentinel_tiles))

    # Find the tiles we haven't downloaded yet
    sentinel_tile_ids = ["_".join(sentinel_tile.split('/')[-1].split('_')[:2]) for sentinel_tile in sentinel_tiles]
    downloaded = [f"{tile_id}_binary_tree_cover_10m.tiff" for tile_id in sentinel_tile_ids]
    df_new = df_2017_2022[~df_2017_2022['tif'].isin(downloaded)]
    print("Number of new tiles to download:", len(df_new))

    rows = df_new[['tif', 'year', 'bbox', 'crs']].values.tolist()
    return rows


# Creating a gpkg with all the sentinel years for each of Nick's tiffs.
filename = '/g/data/xe2/cb8590/Nick_outlines/tiff_footprints_years.gpkg'
gdf = gpd.read_file(filename)
gdf

gdf[gdf['year'] > 2017]
