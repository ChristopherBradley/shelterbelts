import os
import argparse
import glob
import xarray as xr
import rioxarray as rxr
import rasterio

# # +
# # Change directory to this repo - this should work on gadi or locally via python or jupyter.
# import os, sys
# repo_name = "shelterbelts"
# if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
#     repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
# elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
#     repo_dir = os.path.dirname(os.getcwd())       
# else:                                             # Already running from root of this repo. 
#     repo_dir = os.getcwd()
# src_dir = os.path.join(repo_dir, 'src')
# os.chdir(src_dir)
# sys.path.append(src_dir)
# # print(src_dir)

from shelterbelts.apis.worldcover import tif_categorical, worldcover_labels
# -

cmap_woody_veg = {
    0: (240, 240, 240), # Non-trees are white
    1: (0, 100, 0),   # Trees are green
    255: (0, 100, 200)  # Nodata is blue
}
labels_woody_veg = {
    0: "Non-trees",
    1: "Trees",
    255: "Nodata"
}


def worldcover_trees(filename, outdir, da=None, stub=None, savetif=True):
    """Convert a worldcover tif into a binary tree cover tif"""
    # These arguments are unintuitive... I should rethink the arguments you provide this function
    # Probably better to have two functions, one that you supply the filename, outdir and optional stub, and one where you just supply the da.

    if da is None:
        da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

    # Trees or Shrubland
    da_trees = (da == 10) | (da == 20)
    da_trees = da_trees.astype('uint8')
        
    if savetif:
        if stub is None:
            stub = filename.split('/')[-1].split('.')[0]
        outpath = os.path.join(outdir, f"{stub}_woody_veg.tif")
        tif_categorical(da_trees, outpath, cmap_woody_veg, tiled=True)
        
        # Add pyramid for faster viewing in zoomed out views QGIS
        levels  = [2, 4, 8, 16, 32, 64]
        with rasterio.open(outpath, "r+") as src:          
            src.build_overviews(levels)

    ds = da_trees.to_dataset(name='woody_veg')

    # latitude and longitude is what the worldcover API returns and is more readable
    # However, x and y is what rxr.open_rasterio provides, and it's what the rest of the pipeline expects
    ds = ds.rename({'longitude':'x', 'latitude':'y'})

    return ds


def canopy_height_trees(filename, outdir, da=None):
    """Convert a canopy height tif into a binary tree cover tif"""
    if da is None:
        da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

    # Anything taller than 1m
    da_trees = (da >= 1)
    da_trees = da_trees.astype('uint8')

    # Coarsen from 1m to 10m horizontal resolution
    da_trees = da_trees.rio.reproject(
        da_trees.rio.crs,
        resolution=10,                
        resampling=rasterio.enums.Resampling.max  
    )
    
    stub = filename.split('/')[-1].split('.')[0]
    
    outpath = os.path.join(outdir, f"{stub}_woody_veg.tif")
    tif_categorical(da_trees, outpath, cmap_woody_veg, tiled=True)
    
    # Add pyramid for faster viewing in zoomed out views QGIS
    levels  = [2, 4, 8, 16, 32, 64]
    with rasterio.open(outpath, "r+") as src:          
        src.build_overviews(levels)
    
    ds = da_trees.to_dataset(name='woody_veg')
    return ds

    # Trying to avoid memory accumulation
    da.close()          
    da_trees.close()   
    del da, da_trees
    gc.collect()  


# +
funcs = {
    "worldcover_trees": worldcover_trees,
    "canopy_height_trees": canopy_height_trees
}

def run_tifs(folder, func_string, outdir, limit=None):
    """Run the function on every tif file in a folder"""
    func = funcs[func_string]
    tif_files = glob.glob(os.path.join(folder, "*.tif*"))

    if limit:
        tif_files = tif_files[:limit]
    
    for i, tif_file in enumerate(tif_files):
        print(f"{i+1}/{len(tif_files)}: Working on {tif_file}")
        func(tif_file, outdir)


# -

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--folder', help='The folder containing all of the input tiffs')
    parser.add_argument('--func_string', help="Either 'worldcover_trees' or 'canopy_height_trees'")
    parser.add_argument('--outdir', default='.', help='The folder containing all of the output tiffs')
    parser.add_argument('--limit', default=None)

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    if args.limit:
        args.limit = int(args.limit)
    run_tifs(args.folder, args.func_string, args.outdir, args.limit)


