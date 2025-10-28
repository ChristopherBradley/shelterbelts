# # +
# Change directory to this repo. Need to do this when using the DEA environment since I can't just pip install -e .
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
src_dir = os.path.join(repo_dir, 'src')
os.chdir(src_dir)
sys.path.append(src_dir)
# print(src_dir)


# +
# %%time
import os
import glob
import argparse

import geopandas as gpd
import rasterio
import rioxarray as rxr
from shapely.geometry import box      

from shelterbelts.classifications.binary_trees import worldcover_trees, canopy_height_trees
from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.worldcover import worldcover_bbox, tif_categorical
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import canopy_height_bbox, merge_tiles_bbox, merged_ds
from shelterbelts.apis.barra_daily import barra_daily

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics, linear_categories_cmap

# 11 secs for all these imports
# -


worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'  # Should move these to gdata so they don't disappear.
# worldcover_footprints = '/scratch/xe2/cb8590/Worldcover_Australia/Worldcover_Australia_footprints.gpkg'
canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'
hydrolines_gdb = '/g/data/xe2/cb8590/Outlines/SurfaceHydrologyLinesRegional.gdb'
roads_gdb = '/g/data/xe2/cb8590/Outlines/2025_09_National_Roads.gdb'


def run_pipeline_tif(percent_tif, outdir='/scratch/xe2/cb8590/tmp', tmpdir='/scratch/xe2/cb8590/tmp', stub=None, 
                     wind_method=None, wind_threshold=15,
                     cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=10, density_threshold=5, buffer_width=3, strict_core_area=True):
    """Starting from a percent_cover tif, go through the whole pipeline"""
    if stub is None:
        # stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'
        stub = percent_tif.split('/')[-1].split('.')[0][:50] # Hopefully there's something unique in the first 50 characters
    data_folder = percent_tif[percent_tif.find('DATA'):percent_tif.find('DATA') + 11]

    da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')

    gs_bounds = gpd.GeoSeries([box(*da_percent.rio.bounds())], crs=da_percent.rio.crs)
    bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
    worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
    # import pdb; pdb.set_trace()
    
    print("Getting worldcover for tif:", percent_tif)
    mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, f'{data_folder}_{stub}', worldcover_dir, worldcover_geojson, 'filename', verbose=False)     # Need to include the DATA... in the stub so we don't get rasterio merge conflicts
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    gdf_hydrolines, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, da=da_percent)
    gdf_roads, ds_roads = hydrolines(None, roads_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, da=da_percent, layer='NationalRoads_2025_09')

    if wind_method: 
        lat = (bbox_4326[1] + bbox_4326[3])/2
        lon = (bbox_4326[0] + bbox_4326[2])/2
        ds_wind = barra_daily(lat=lat, lon=lon, start_year=2020, end_year=2020, gdata=True, plot=False, save_netcdf=False) # This line is currently the limiting factor since it takes 4 secs
    else:
        # if no wind_method provided than the percent_cover method without wind gets used
        ds_wind = None

    da_trees = da_percent > cover_threshold
    ds_woody_veg = da_trees.to_dataset(name='woody_veg')
    ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=min_patch_size, edge_size=edge_size, max_gap_size=max_gap_size, strict_core_area=strict_core_area, save_tif=False, plot=False, ds=ds_woody_veg)
    ds_shelter = shelter_categories(None, wind_method=wind_method, wind_threshold=wind_threshold, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories, ds_wind=ds_wind)
    ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=False, plot=False, da_worldcover=da_worldcover)

    ds_buffer = buffer_categories(None, None, buffer_width=buffer_width, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines, ds_roads=ds_roads)
    ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False) 
    
    return ds_linear


def run_pipeline_tifs(folder, outdir='/scratch/xe2/cb8590/tmp', tmpdir='/scratch/xe2/cb8590/tmp', param_stub='', 
                      wind_method=None, wind_threshold=15,
                      cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
                      distance_threshold=10, density_threshold=5, buffer_width=3, strict_core_area=False):
    """
    Starting from a folder of percent_cover tifs, go through the whole shelterbelt delineation pipeline

    Parameters
    ----------
        folder: The input folder with the percent_cover.tifs.
        outdir: The folder to save the output linear_categories.tifs.
        tmpdir: A folder where it's ok to save temporary files to be deleted later.
        cover_threshold: Percentage tree cover within a 10m pixel to be classified as a boolean 'tree'.
        min_patch_size: The minimum area to be classified as a patch/corrider rather than just scattered trees.
        edge_size: The buffer distance at the edge of a patch, with pixels inside this being the 'core area'. 
        max_gap_size: The allowable gap between two tree clusters before considering them separate patches.
        distance_threshold: The distance from trees that counts as sheltered.
        density_threshold: The percentage tree cover within a radius of distance_threshold that counts as sheltered.
        buffer_width: Number of pixels away from the feature that still counts as being within the buffer.

    Downloads
    ---------
        merged.tif: A combined tif after applying the full pipeline to every individual tif
    
    
    """
    os.makedirs(outdir, exist_ok=True)
    percent_tifs = glob.glob(f'{folder}/*.tif')
    for percent_tif in percent_tifs:
        run_pipeline_tif(percent_tif, outdir, tmpdir, None, wind_method, wind_threshold, cover_threshold, min_patch_size, edge_size, max_gap_size, distance_threshold, density_threshold, buffer_width, strict_core_area)
    gdf = bounding_boxes(outdir)
    stub = '_'.join(outdir.split('/')[-2:]).split('.')[0]  # The filename and one folder above
    
    footprint_gpkg = f"{stub}_footprints.gpkg"
    bbox =[gdf.bounds['minx'].min(), gdf.bounds['miny'].min(), gdf.bounds['maxx'].max(), gdf.bounds['maxy'].max()]
    mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, stub, outdir, footprint_gpkg, id_column='filename')  
    ds = merged_ds(mosaic, out_meta, 'linear_categories')

    basedir = os.path.dirname(outdir)
    filename_linear = os.path.join(basedir, f'{stub}_merged_{param_stub}.tif')
    tif_categorical(ds['linear_categories'], filename_linear, linear_categories_cmap) 
    return ds


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the shelterbelt delineation pipeline on a folder of percent_cover.tifs.")

    parser.add_argument("folder", help="Input folder containing percent_cover.tifs")
    parser.add_argument("--outdir", default="/scratch/xe2/cb8590/tmp", help="Output folder for linear_categories.tifs (default: /scratch/xe2/cb8590/tmp)")
    parser.add_argument("--tmpdir", default="/scratch/xe2/cb8590/tmp", help="Temporary working folder (default: /scratch/xe2/cb8590/tmp)")
    parser.add_argument("--param_stub", default="", help="Extra stub for the suffix of the merged tif")  # Don't need this argument anymore, better to just incorporate these parameter names in the outdir
    parser.add_argument("--wind_method", default=None, help="Method to use to determine shelter direction")
    parser.add_argument("--wind_threshold", default=15, help="Windspeed that causes damage to crops/pasture in km/hr (default: 15)")
    parser.add_argument("--cover_threshold", type=int, default=10, help="Percentage tree cover within a pixel to classify as tree (default: 10)")
    parser.add_argument("--min_patch_size", type=int, default=20, help="Minimum patch size in pixels (default: 20)")
    parser.add_argument("--edge_size", type=int, default=3, help="Buffer distance at patch edges for core area (default: 3)")
    parser.add_argument("--max_gap_size", type=int, default=1, help="Maximum gap between tree clusters (default: 1)")
    parser.add_argument("--distance_threshold", type=int, default=10, help="Distance from trees that counts as sheltered (default: 10)")
    parser.add_argument("--density_threshold", type=int, default=5, help="Tree cover %% within distance_threshold that counts as sheltered (default: 5)")
    parser.add_argument("--buffer_width", type=int, default=3, help="Buffer width for sheltered area (default: 3)")
    parser.add_argument("--strict_core_area", default=False, action="store_true", help="Boolean to determine whether to enforce core areas to be fully connected.")

    return parser.parse_args()



# +
if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline_tifs(
        folder=args.folder,
        outdir=args.outdir,
        tmpdir=args.tmpdir,
        param_stub=args.param_stub,
        wind_method=args.wind_method,
        wind_threshold=args.wind_threshold,
        cover_threshold=args.cover_threshold,
        min_patch_size=args.min_patch_size,
        edge_size=args.edge_size,
        max_gap_size=args.max_gap_size,
        distance_threshold=args.distance_threshold,
        density_threshold=args.density_threshold,
        buffer_width=args.buffer_width,
        strict_core_area=args.strict_core_area
    )

# # +
# # %%time
# cover_threshold=0
# min_patch_size=20
# edge_size=3
# max_gap_size=1
# distance_threshold=10
# density_threshold=5 
# buffer_width=3
# strict_core_area=False
# param_stub = ""
# wind_method=None
# wind_threshold=15
# # folder = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717840/uint8_percentcover_res10_height2m/'
# # outdir = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717840/linear_tifs'

# folder='/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_28_lon_142'
# outdir='/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/linear_tifs_lat_28_lon_142'
# tmpdir = '/scratch/xe2/cb8590/tmp'
# # -


# # %%time
# run_pipeline_tifs(folder, outdir, tmpdir, param_stub='riparian_doubleoverride')


# +
# # %%time
# # Single tif example for debugging
# # percent_tif = '/scratch/xe2/cb8590/lidar/DATA_722798/uint8_percentcover_res10_height2m/Wellington201409-PHO3-C0-AHD_6666384_55_0002_0002_percentcover_res10_height2m_uint8.tif'
# # percent_tif = '/scratch/xe2/cb8590/ACTGOV_my_processing/uint8_percentcover_res10_height2m/ACT-16ppm_2025_SW_679000_6099000_1k_class_AHD_percentcover_res10_height2m_uint8.tif'
# # percent_tif = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717840/uint8_percentcover_res10_height2m/Young201709-LID1-C3-AHD_6306194_55_0002_0002_percentcover_res10_height2m_uint8.tif'
# percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_28_lon_142/29_21-143_98_y2024_predicted.tif'
# stub = None
# if stub is None:
#     # stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'
#     stub = percent_tif.split('/')[-1].split('.')[0][:50] # Hopefully there's something unique in the first 50 characters
# data_folder = percent_tif[percent_tif.find('DATA'):percent_tif.find('DATA') + 11]

# da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')

# gs_bounds = gpd.GeoSeries([box(*da_percent.rio.bounds())], crs=da_percent.rio.crs)
# bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
# worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
# # import pdb; pdb.set_trace()
# -


# mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, f'{data_folder}_{stub}', worldcover_dir, worldcover_geojson, 'filename', verbose=False)     # Need to include the DATA... in the stub so we don't get rasterio merge conflicts


# +
# ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
# da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
# gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, da=da_percent)
# gdf_roads, ds_roads = hydrolines(None, roads_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=False, da=da_percent, layer='NationalRoads_2025_09')

# lat = (bbox_4326[1] + bbox_4326[3])/2
# lon = (bbox_4326[0] + bbox_4326[2])/2
# ds_wind = barra_daily(lat=lat, lon=lon, start_year=2020, end_year=2020, gdata=True, plot=False, save_netcdf=False) # This line is currently the limiting factor since it takes 4 secs

# da_trees = da_percent > cover_threshold
# ds_woody_veg = da_trees.to_dataset(name='woody_veg')
# ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=min_patch_size, edge_size=edge_size, max_gap_size=max_gap_size, save_tif=False, plot=False, ds=ds_woody_veg)
# # ds_shelter = shelter_categories(None, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories)  # percent treecover method
# ds_shelter = shelter_categories(None, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories, ds_wind=ds_wind)

# ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=False, plot=False, da_worldcover=da_worldcover)

# ds_buffer = buffer_categories(None, None, buffer_width=buffer_width, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines, ds_roads=ds_roads)
# ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False) 

# +
# from shelterbelts.indices.shelter_metrics import linear_categories_labels
# {item[0]:item[1] for item in sorted(linear_categories_labels.items())}
