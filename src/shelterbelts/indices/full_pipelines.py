import os
import glob
import argparse
import math
import pathlib

import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box

# Trying to avoid memory issues
import gc
import psutil
import subprocess, sys

repo_name = "shelterbelts"
import sys, os
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
    src_dir = os.path.join(repo_dir, 'src')
    os.chdir(src_dir)
    sys.path.append(src_dir)
    # print(src_dir)

from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.utils.visualization import tif_categorical
from shelterbelts.utils.crop_and_rasterize import crop_and_rasterize
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds
from shelterbelts.apis.barra_daily import barra_daily

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import patch_metrics, linear_categories_cmap

# 11 secs for all these imports
# -
from shelterbelts.utils.filepaths import (
    worldcover_dir,
    worldcover_geojson,
    hydrolines_gdb,
    roads_gdb,
    default_outdir,
    default_tmpdir
)

process = psutil.Process(os.getpid())


def run_pipeline_tif(percent_tif, outdir=default_outdir,
                     tmpdir=default_tmpdir, stub=None,
                     wind_method=None, wind_threshold=20,
                     cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                     crop_pixels=0, min_core_size=1000, min_shelterbelt_length=20, max_shelterbelt_width=6,
                     worldcover_dir=worldcover_dir, worldcover_geojson=worldcover_geojson, 
                     hydrolines_gdb=hydrolines_gdb, roads_gdb=roads_gdb):
    """Starting from a percent_cover tif, go through the whole pipeline
    
    Parameters
    ----------
        percent_tif: Path to input percent cover tif
        worldcover_dir: Directory containing worldcover data (defaults to NCI path)
        worldcover_geojson: Filename of worldcover footprints (defaults to NCI filename)
        hydrolines_gdb: Path to hydrolines geodatabase (defaults to NCI path)
        roads_gdb: Path to roads geodatabase (defaults to NCI path)
    """
    if stub is None:
        # stub = "_".join(percent_tif.split('/')[-1].split('.')[0].split('_')[:2])  # e.g. 'Junee201502-PHO3-C0-AHD_5906174'
        stub = percent_tif.split('/')[-1].split('.')[0][:50] # Hopefully there's something unique in the first 50 characters
    # Extract data_folder from ELVIS filenaming system, or use a generic stub if not found
    data_folder_idx = percent_tif.find('DATA')
    if data_folder_idx != -1:
        data_folder = percent_tif[data_folder_idx:data_folder_idx + 11]
    else:
        data_folder = 'generic'

    da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
    da_trees = da_percent > cover_threshold

    gs_bounds = gpd.GeoSeries([box(*da_percent.rio.bounds())], crs=da_percent.rio.crs)
    bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
    
    # import pdb; pdb.set_trace()
    # Anything that might be run in parallel needs a unique filename, so we don't get rasterio merge conflicts
    worldcover_stub = f'{data_folder}_{stub}_{wind_method}_w{wind_threshold}_c{cover_threshold}_m{min_patch_size}_e{edge_size}_g{max_gap_size}_di{distance_threshold}_de{density_threshold}_b{buffer_width}_mc{min_core_size}_msl{min_shelterbelt_length}_msw{max_shelterbelt_width}_sca{strict_core_area}' # 
    
    mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, worldcover_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    gdf_hydrolines, ds_hydrolines = crop_and_rasterize(da_percent, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, feature_name='gullies')
    gdf_roads, ds_roads = crop_and_rasterize(da_percent, roads_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, layer='NationalRoads_2025_09', feature_name='roads')

    if wind_method and wind_method != "None":  # Handling conversion of None to "None" when using subprocess
        lat = (bbox_4326[1] + bbox_4326[3])/2
        lon = (bbox_4326[0] + bbox_4326[2])/2
        ds_wind = barra_daily(lat=lat, lon=lon, start_year=2020, end_year=2020, gdata=True, plot=False, save_netcdf=False) # This line is currently the limiting factor since it takes 4 secs
    else:
        # if no wind_method provided than the percent_cover method without wind gets used
        ds_wind = None

    ds_woody_veg = da_trees.to_dataset(name='woody_veg')
    ds_tree_categories = tree_categories(ds_woody_veg, outdir, stub, min_patch_size=min_patch_size, min_core_size=min_core_size, edge_size=edge_size, max_gap_size=max_gap_size, strict_core_area=strict_core_area, save_tif=False, plot=False)
    ds_shelter = shelter_categories(ds_tree_categories, wind_data=ds_wind, wind_method=wind_method, wind_threshold=wind_threshold, distance_threshold=distance_threshold, density_threshold=density_threshold, outdir=outdir, stub=stub, savetif=False, plot=False, crop_pixels=crop_pixels)
    ds_cover = cover_categories(ds_shelter, da_worldcover, outdir=outdir, stub=stub, savetif=False, plot=False)

    ds_buffer = buffer_categories(ds_cover, ds_hydrolines, roads_data=ds_roads, outdir=outdir, stub=stub, buffer_width=buffer_width, savetif=False, plot=False)
    ds_linear, df_patches = patch_metrics(ds_buffer, outdir, stub, plot=False, save_csv=False, save_labels=False, crop_pixels=crop_pixels, min_shelterbelt_length=min_shelterbelt_length, max_shelterbelt_width=max_shelterbelt_width) 

    # Trying to avoid memory accumulation
    for ds in [ds_worldcover, ds_roads, ds_hydrolines, ds_woody_veg, ds_tree_categories, ds_shelter, ds_cover, ds_buffer, ds_linear]:
        try:
            ds.close()
            del ds
        except Exception:
            pass
    del df_patches
    locals().clear()
    gc.collect()
    # rasterio.shutil.delete_raster_cache()
    mem_info = process.memory_full_info()
    # print(f"RSS: {mem_info.rss / 1e9:.2f} GB, VMS: {mem_info.vms / 1e9:.2f} GB, Shared: {mem_info.shared / 1e9:.2f} GB")
    # print("Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "MB")
    # print("Number of open files:", len(psutil.Process(os.getpid()).open_files()))
    return None

def run_pipeline_csv(csv, outdir=default_outdir,
                     tmpdir=default_tmpdir, stub=None,
                     wind_method=None, wind_threshold=20,
                     cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
                     distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                     crop_pixels=0, min_core_size=1000, min_shelterbelt_length=20, max_shelterbelt_width=6,
                     worldcover_dir=worldcover_dir, worldcover_geojson=worldcover_geojson,
                     hydrolines_gdb=hydrolines_gdb, roads_gdb=roads_gdb):
    """Run the pipeline for every tif in a csv"""
    df = pd.read_csv(csv)
    for percent_tif in df['filename']:
        # The provided stub needs to be None, because we want to use the percent_tif filename instead. 
        run_pipeline_tif(percent_tif, outdir, tmpdir, None, wind_method, wind_threshold, cover_threshold, min_patch_size, edge_size, max_gap_size, distance_threshold, density_threshold, buffer_width, strict_core_area, crop_pixels, min_core_size, min_shelterbelt_length, max_shelterbelt_width, worldcover_dir, worldcover_geojson, hydrolines_gdb, roads_gdb)


def run_pipeline_tifs(folder, outdir=default_outdir, tmpdir=default_tmpdir, param_stub='', 
                      wind_method=None, wind_threshold=20,
                      cover_threshold=10, min_patch_size=20, edge_size=3, max_gap_size=1,
                      distance_threshold=20, density_threshold=5, buffer_width=3, strict_core_area=True,
                      crop_pixels=0, limit=None, tiles_per_csv=100, min_core_size=1000, min_shelterbelt_length=20, max_shelterbelt_width=6, merge_outputs=False, suffix='tif',
                      worldcover_dir=worldcover_dir, worldcover_geojson=worldcover_geojson,
                      hydrolines_gdb=hydrolines_gdb, roads_gdb=roads_gdb):
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
    percent_tifs = glob.glob(f'{folder}/*.{suffix}')
    print(f"Starting with {len(percent_tifs)} percent_tifs", flush=True)

    if limit:
        percent_tifs = percent_tifs[:limit]

    if limit is None: # Don't remove tifs if we've specified a limit, because it's just for testing so I want reproducible results.
        # Remove tifs that have already been processed (sometimes I have to run this multiple times if a process runs out of memory or rasterio gives a parallelisation conflict)
        percent_stubs = [pathlib.Path(tif).stem[:12] for tif in percent_tifs]
        processed = glob.glob(f'{outdir}/*.tif')
        processed_stubs = set(pathlib.Path(tif).stem[:12] for tif in processed)
        percent_tifs = [tif for tif, stub in zip(percent_tifs, percent_stubs) if stub not in processed_stubs]
        print(f"Reduced to {len(percent_tifs)} percent_tifs", flush=True)

    df = pd.DataFrame(percent_tifs, columns=["filename"])
    csv_filenames = []
    chunk_size = tiles_per_csv
    for i in range(math.ceil(len(df) / chunk_size)):
        chunk = df[i*chunk_size : (i+1)*chunk_size]
        all_the_params = f'{wind_method}_w{wind_threshold}_c{cover_threshold}_m{min_patch_size}_e{edge_size}_g{max_gap_size}_di{distance_threshold}_de{density_threshold}_b{buffer_width}_mc{min_core_size}_msl{min_shelterbelt_length}_msw{max_shelterbelt_width}_sca{strict_core_area}' # Anything that might be run in parallel needs a unique filename
        filename = os.path.join(tmpdir, f"{param_stub}_{all_the_params}_run_pipeline_tifs_{i}.csv")
        chunk.to_csv(filename, index=False)
        csv_filenames.append(filename)
        print("Saved:", filename)

    for i, filename in enumerate(csv_filenames):
        print(f"Launching Popen subprocess for filename {i}/{len(csv_filenames)}:", filename)

        cmd = [
            sys.executable,
            # "full_pipelines.py", 
            "shelterbelts/indices/full_pipelines.py",  
            str(filename),
            "--outdir", str(outdir),
            "--tmpdir", str(tmpdir),
            "--param_stub", str(param_stub),  # or args.param_stub if applicable
            "--wind_method", str(wind_method),
            "--wind_threshold", str(wind_threshold),
            "--cover_threshold", str(cover_threshold),
            "--min_patch_size", str(min_patch_size),
            "--edge_size", str(edge_size),
            "--max_gap_size", str(max_gap_size),
            "--distance_threshold", str(distance_threshold),
            "--density_threshold", str(density_threshold),
            "--buffer_width", str(buffer_width),
            "--crop_pixels", str(crop_pixels),
            "--min_core_size", str(min_core_size),
            "--min_shelterbelt_length", str(min_shelterbelt_length),
            "--max_shelterbelt_width", str(max_shelterbelt_width),
            "--worldcover_dir", str(worldcover_dir) if worldcover_dir else "",
            "--worldcover_geojson", str(worldcover_geojson) if worldcover_geojson else "",
            "--hydrolines_gdb", str(hydrolines_gdb) if hydrolines_gdb else "",
            "--roads_gdb", str(roads_gdb) if roads_gdb else ""
        ]
        if strict_core_area:
            cmd += ["--strict_core_area"]
        
        # Popen a subprocess to hopefully avoid memory accumulation
        p = subprocess.Popen(cmd)
        p.wait()

    if merge_outputs:
        # Merge the outputs
        if wind_method:
            suffix_stems = ['linear_categories', 'distances']
        else:
            suffix_stems = ['linear_categories', 'densities']
        for suffix_stem in suffix_stems:
            filetype=f'{suffix_stem}.tif'
            stub_original = f"{'_'.join(folder.split('/')[-2:]).split('.')[0]}_{suffix_stem}"  # The filename and one folder above with the suffix. 
            
            stub = f'{stub_original}_{wind_method}_w{wind_threshold}_c{cover_threshold}_m{min_patch_size}_e{edge_size}_g{max_gap_size}_di{distance_threshold}_de{density_threshold}_b{buffer_width}_mc{min_core_size}_msl{min_shelterbelt_length}_msw{max_shelterbelt_width}_sca{strict_core_area}' # Anything that might be run in parallel needs a unique filename, so we don't get rasterio merge conflicts
            gdf = bounding_boxes(outdir, stub=stub, filetype=filetype, verbose=False)  # Exclude the shelter_distances.tif from the merging. Need to include this filetype in the gpkg name so I can merge the densities/distances too. 
            
            footprint_gpkg = f"{stub}_footprints.gpkg"
            bbox =[gdf.bounds['minx'].min(), gdf.bounds['miny'].min(), gdf.bounds['maxx'].max(), gdf.bounds['maxy'].max()]
            mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, stub, outdir, footprint_gpkg, id_column='filename', verbose=False)  
            ds = merged_ds(mosaic, out_meta, suffix_stem)
            basedir = os.path.dirname(outdir)
            
            filename_linear = os.path.join(basedir, f'{stub}_merged_{param_stub}.tif')
            tif_categorical(ds[suffix_stem], filename_linear, linear_categories_cmap) # The distances and densities should use a continuous cmap ranging from 0-100 instead
        return ds


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the shelterbelt delineation pipeline on a folder of percent_cover.tifs.")

    parser.add_argument("folder", help="Input folder containing percent_cover.tifs")
    parser.add_argument("--outdir", default=default_outdir, help=f"Output folder for linear_categories.tifs (default: {default_outdir})")
    parser.add_argument("--tmpdir", default=default_tmpdir, help=f"Temporary working folder (default: {default_tmpdir})")
    parser.add_argument("--param_stub", default=None, help="Extra stub for the suffix of the merged tif")
    parser.add_argument("--wind_method", default=None, help="Method to use to determine shelter direction")
    parser.add_argument("--wind_threshold", type=int, default=20, help="Windspeed that causes damage to crops/pasture in km/hr (default: 20)")
    parser.add_argument("--cover_threshold", type=int, default=10, help="Percentage tree cover within a pixel to classify as tree (default: 10)")
    parser.add_argument("--min_patch_size", type=int, default=20, help="Minimum patch size in pixels (default: 20)")
    parser.add_argument("--edge_size", type=int, default=3, help="Buffer distance at patch edges for core area (default: 3)")
    parser.add_argument("--max_gap_size", type=int, default=1, help="Maximum gap between tree clusters (default: 1)")
    parser.add_argument("--distance_threshold", type=int, default=20, help="Distance from trees that counts as sheltered (default: 20)")
    parser.add_argument("--density_threshold", type=int, default=5, help="Tree cover %% within distance_threshold that counts as sheltered (default: 5)")
    parser.add_argument("--buffer_width", type=int, default=3, help="Buffer width for sheltered area (default: 3)")
    parser.add_argument("--crop_pixels", type=int, default=0, help="Number of pixels to crop from the linear_tif (default: 0)")
    parser.add_argument("--strict_core_area", default=False, action="store_true", help="Boolean to determine whether to enforce core areas to be fully connected.")
    parser.add_argument("--limit", type=int, default=None, help="Number of tifs to process (default: all)")
    parser.add_argument("--min_core_size", type=int, default=1000, help="The minimum area to be classified as a core, rather than just a patch or corridor. (default: 100)")
    parser.add_argument("--min_shelterbelt_length", type=int, default=20, help="The minimum length to be classified as a shelterbelt. (default: 20)")
    parser.add_argument("--max_shelterbelt_width", type=int, default=6, help="The maximum average width to be classified as a shelterbelt. (default: 4)")
    parser.add_argument("--suffix", default='tif', help="Suffix of each of the input tif files")
    parser.add_argument("--worldcover_dir", default=None, help="Directory containing worldcover data")
    parser.add_argument("--worldcover_geojson", default=None, help="Filename of worldcover footprints")
    parser.add_argument("--hydrolines_gdb", default=None, help="Path to hydrolines geodatabase")
    parser.add_argument("--roads_gdb", default=None, help="Path to roads geodatabase")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.folder.endswith('.tif'):
        run_pipeline_tif(
            args.folder,
            outdir=args.outdir,
            tmpdir=args.tmpdir,
            stub=args.param_stub,
            wind_method=args.wind_method,
            wind_threshold=args.wind_threshold,
            cover_threshold=args.cover_threshold,
            min_patch_size=args.min_patch_size,
            edge_size=args.edge_size,
            max_gap_size=args.max_gap_size,
            distance_threshold=args.distance_threshold,
            density_threshold=args.density_threshold,
            buffer_width=args.buffer_width,
            strict_core_area=args.strict_core_area,
            crop_pixels=args.crop_pixels,
            min_core_size=args.min_core_size,
            min_shelterbelt_length=args.min_shelterbelt_length,
            max_shelterbelt_width=args.max_shelterbelt_width,
            worldcover_dir=args.worldcover_dir,
            worldcover_geojson=args.worldcover_geojson,
            hydrolines_gdb=args.hydrolines_gdb,
            roads_gdb=args.roads_gdb
        )
    elif args.folder.endswith('.csv'):
            run_pipeline_csv(
            args.folder,
            outdir=args.outdir,
            tmpdir=args.tmpdir,
            stub=args.param_stub,
            wind_method=args.wind_method,
            wind_threshold=args.wind_threshold,
            cover_threshold=args.cover_threshold,
            min_patch_size=args.min_patch_size,
            edge_size=args.edge_size,
            max_gap_size=args.max_gap_size,
            distance_threshold=args.distance_threshold,
            density_threshold=args.density_threshold,
            buffer_width=args.buffer_width,
            strict_core_area=args.strict_core_area,
            crop_pixels=args.crop_pixels,
            min_core_size=args.min_core_size,
            min_shelterbelt_length=args.min_shelterbelt_length,
            max_shelterbelt_width=args.max_shelterbelt_width,
            worldcover_dir=args.worldcover_dir,
            worldcover_geojson=args.worldcover_geojson,
            hydrolines_gdb=args.hydrolines_gdb,
            roads_gdb=args.roads_gdb
        )
    else:
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
            strict_core_area=args.strict_core_area,
            crop_pixels=args.crop_pixels,
            limit=args.limit,
            min_core_size=args.min_core_size,
            min_shelterbelt_length=args.min_shelterbelt_length,
            max_shelterbelt_width=args.max_shelterbelt_width,
            suffix=args.suffix,
            worldcover_dir=args.worldcover_dir,
            worldcover_geojson=args.worldcover_geojson,
            hydrolines_gdb=args.hydrolines_gdb,
            roads_gdb=args.roads_gdb
        )