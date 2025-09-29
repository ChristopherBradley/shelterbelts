# +
# %%time
import os
import glob

import geopandas as gpd
import rasterio
import rioxarray as rxr
from shapely.geometry import box      

from shelterbelts.classifications.binary_trees import worldcover_trees, canopy_height_trees
from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.apis.worldcover import worldcover_bbox, tif_categorical
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import canopy_height_bbox, merge_tiles_bbox, merged_ds

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics, linear_categories_cmap

# 11 secs for all these imports


# +
worldcover_dir = '/scratch/xe2/cb8590/Worldcover_Australia'  # Should move these to gdata so they don't disappear.
# worldcover_footprints = '/scratch/xe2/cb8590/Worldcover_Australia/Worldcover_Australia_footprints.gpkg'
canopy_height_dir = '/scratch/xe2/cb8590/Global_Canopy_Height'
hydrolines_gdb = '/g/data/xe2/cb8590/Outlines/SurfaceHydrologyLinesRegional.gdb'

def run_pipeline(bbox, outdir, stub):
    """Starting from a bbox, go through the whole pipeline"""

    # Load worldcover, canopy height and hydrolines
    mosaic, out_meta = merge_tiles_bbox(bbox, outdir, stub, worldcover_dir)
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    ds_canopy_height = canopy_height_bbox(bbox, outdir=outdir, stub=stub, tmpdir=canopy_height_dir, save_tif=False, plot=False, footprints_geojson='tiles_global.geojson')
    gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=".", stub="TEST", savetif=False, save_gpkg=False, da=da_worldcover)

    # Use the canopy height for starting trees (replace this with my predictions later)
    ds_woody_veg = canopy_height_trees(None, savetif=False, da=ds_canopy_height['canopy_height'])

    # Rest of the pipeline (play around with parameters and edit the functions more later)
    ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=20, edge_size=3, max_gap_size=1, save_tif=False, plot=False, ds=ds_woody_veg)
    ds_shelter = shelter_categories(None, distance_threshold=10, density_threshold=5, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories)
    ds_cover = cover_categories(None, None, outdir=outdir, stub=stub, ds=ds_shelter, savetif=False, plot=False, da_worldcover=da_worldcover)
    ds_buffer = buffer_categories(None, None, buffer_width=3, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_cover, ds_gullies=ds_hydrolines)
    ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False)




# +
# # # Creating a bounding box for worldcover
# gs_bounds = gpd.GeoSeries([box(*da.rio.bounds())], crs=da.rio.crs)
# bbox_3857 = list(gs_bounds.to_crs('EPSG:3857').bounds.iloc[0])
# bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])

# # mosaic, out_meta = merge_tiles_bbox(bbox, outdir, stub, worldcover_dir, worldcover_footprints, 'filename')
# # ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
# # da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})

# # # Don't need this for my ground truth shelterbelts since I can use the lidar directly
# # ds_canopy_height = canopy_height_bbox(bbox_4326, outdir=outdir, stub=stub, tmpdir=canopy_height_dir, save_tif=False, plot=False, footprints_geojson='tiles_global.geojson')
# # ds_canopy_height['canopy_height']

# # # %%time
# # ds_woody_veg = canopy_height_trees(None, savetif=False, da=da_chm)  # Actually takes a while to do this interpolation (20 secs), so better off to just use the pre-interpolated percent trees
# -

def example_pipeline():
    # An old example of running the pipeline from the command line
    filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_10.gpkg'
    outdir = '/scratch/xe2/cb8590/tmp4'
    gdf_barra_bboxs = gpd.read_file(filename)
    
    # Choose a bbox
    bbox_polygon = gdf_barra_bboxs.iloc[0]
    bbox = bbox_polygon['geometry'].bounds
    
    # Create a stub
    centroid = bbox_polygon.geometry.centroid
    stub = f"{centroid.y:.2f}-{centroid.x:.2f}".replace(".", "_")[1:]
    
    for i, row in gdf_barra_bboxs.iterrows():
        bbox = row['geometry'].bounds
        centroid = row['geometry'].centroid
        stub = f"{centroid.y:.2f}-{centroid.x:.2f}".replace(".", "_")[1:]
        run_pipeline(bbox, outdir, stub) 
    
    # Takes about 4 secs per tile
    # So should take about an hour per 1000 tiles. 

# +
# def example2(percent_tif, chm_tif, threshold=10, outdir='/scratch/xe2/cb8590/tmp', stub=None,
#                      min_patch_size=20, edge_size=3, max_gap_size=1,
#                      distance_threshold=10, density_threshold=5, buffer_width=3):
#     """Starting from a percent_cover tif, go through the whole pipeline"""
#     if stub is None:
#         stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'

#     da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')

#     gs_bounds = gpd.GeoSeries([box(*da_percent.rio.bounds())], crs=da.rio.crs)
#     bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
    
#     # The chm_tif is only relevant if I change the wind_direction method in ds_shelter
#     if chm_tif is None:
#         ds_canopy_height = canopy_height_bbox(bbox_4326, outdir=outdir, stub=stub, tmpdir=canopy_height_dir, save_tif=False, plot=False, footprints_geojson='tiles_global.geojson')
#         da_chm = ds_canopy_height['canopy_height']
#     else:
#         da_chm = rxr.open_rasterio(chm_tif).isel(band=0).drop_vars('band')
        
#     gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=".", stub="TEST", savetif=False, save_gpkg=False, da=da_percent)
#     da_trees = da_percent > cover_threshold
#     ds_woody_veg = da_trees.to_dataset(name='woody_veg')
    
#     ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=min_patch_size, edge_size=edge_size, max_gap_size=max_gap_size, save_tif=False, plot=False, ds=ds_woody_veg)
#     ds_shelter = shelter_categories(None, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories)
#     ds_shelter['cover_categories'] = ds_shelter['shelter_categories']  # Skipping the worldcover for now
#     ds_buffer = buffer_categories(None, None, buffer_width=buffer_width, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_shelter, ds_gullies=ds_hydrolines)
#     ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False) 
    
#     return ds_linear


# -


def run_pipeline_tif(percent_tif, outdir='/scratch/xe2/cb8590/tmp', tmpdir='/scratch/xe2/cb8590/tmp', stub=None, 
                     cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=10, density_threshold=5, buffer_width=3):
    """Starting from a percent_cover tif, go through the whole pipeline"""
    if stub is None:
        stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'

    da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')

    gs_bounds = gpd.GeoSeries([box(*da_percent.rio.bounds())], crs=da_percent.rio.crs)
    bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
    worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
    mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False)
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    gdf, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=".", stub="TEST", savetif=False, save_gpkg=False, da=da_percent)
    
    da_trees = da_percent > cover_threshold
    ds_woody_veg = da_trees.to_dataset(name='woody_veg')
    ds_tree_categories = tree_categories(None, outdir, stub, min_patch_size=min_patch_size, edge_size=edge_size, max_gap_size=max_gap_size, save_tif=False, plot=False, ds=ds_woody_veg)
    ds_shelter = shelter_categories(None, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_tree_categories)
    ds_shelter['cover_categories'] = ds_shelter['shelter_categories']  # Skipping the worldcover for now
    ds_buffer = buffer_categories(None, None, buffer_width=buffer_width, outdir=outdir, stub=stub, savetif=False, plot=False, ds=ds_shelter, ds_gullies=ds_hydrolines)
    ds_linear, df_patches = patch_metrics(None, outdir, stub, ds=ds_buffer, plot=False, save_csv=False, save_labels=False) 
    
    return ds_linear


def run_pipeline_tifs(folder, outdir='/scratch/xe2/cb8590/tmp', tmpdir='/scratch/xe2/cb8590/tmp', cover_threshold=10,
                     min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=10, density_threshold=5, buffer_width=3):
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
        run_pipeline_tif(percent_tif, outdir=outdir, tmpdir=tmpdir)
    gdf = bounding_boxes(outdir)
    basedir = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717827'
    stub = '_'.join(outdir.split('/')[-2:]).split('.')[0]  # The filename and one folder above
    
    footprint_gpkg = f"{stub}_footprints.gpkg"
    bbox =[gdf.bounds['minx'].min(), gdf.bounds['miny'].min(), gdf.bounds['maxx'].max(), gdf.bounds['maxy'].max()]
    mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, stub, outdir, footprint_gpkg, id_column='filename')  
    ds = merged_ds(mosaic, out_meta, 'linear_categories')

    basedir = os.path.dirname(folder)
    filename_linear = os.path.join(basedir, f'{stub}_merged.tif')
    tif_categorical(ds['linear_categories'], filename_linear, linear_categories_cmap) 
    return ds



# +
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the shelterbelt delineation pipeline on a folder of percent_cover.tifs."
    )
    parser.add_argument(
        "--folder", required=True, help="Input folder containing percent_cover.tifs"
    )
    parser.add_argument(
        "--outdir", default="/scratch/xe2/cb8590/tmp",
        help="Output folder for linear_categories.tifs (default: /scratch/xe2/cb8590/tmp)"
    )
    parser.add_argument(
        "--tmpdir", default="/scratch/xe2/cb8590/tmp",
        help="Temporary working folder (default: /scratch/xe2/cb8590/tmp)"
    )
    parser.add_argument(
        "--cover_threshold", type=int, default=10,
        help="Percentage tree cover within a pixel to classify as tree (default: 10)"
    )
    parser.add_argument(
        "--min_patch_size", type=int, default=20,
        help="Minimum patch size in pixels (default: 20)"
    )
    parser.add_argument(
        "--edge_size", type=int, default=3,
        help="Buffer distance at patch edges for core area (default: 3)"
    )
    parser.add_argument(
        "--max_gap_size", type=int, default=1,
        help="Maximum gap between tree clusters (default: 1)"
    )
    parser.add_argument(
        "--distance_threshold", type=int, default=10,
        help="Distance from trees that counts as sheltered (default: 10)"
    )
    parser.add_argument(
        "--density_threshold", type=int, default=5,
        help="Tree cover % within distance_threshold that counts as sheltered (default: 5)"
    )
    parser.add_argument(
        "--buffer_width", type=int, default=3,
        help="Buffer width for sheltered area (default: 3)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline_tifs(
        folder=args.folder,
        outdir=args.outdir,
        tmpdir=args.tmpdir,
        cover_threshold=args.cover_threshold,
        min_patch_size=args.min_patch_size,
        edge_size=args.edge_size,
        max_gap_size=args.max_gap_size,
        distance_threshold=args.distance_threshold,
        density_threshold=args.density_threshold,
        buffer_width=args.buffer_width,
    )

# +
# # %%time
# cover_threshold=10
# min_patch_size=20
# edge_size=3
# max_gap_size=1
# distance_threshold=10
# density_threshold=5 
# buffer_width=3
# folder = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717827/uint8_percentcover_res10_height2m'
# outdir = '/scratch/xe2/cb8590/lidar_30km_old/DATA_717827/linear_tifs'
# tmpdir = '/scratch/xe2/cb8590/tmp'
# run_pipeline_tifs(folder, outdir, tmpdir)
