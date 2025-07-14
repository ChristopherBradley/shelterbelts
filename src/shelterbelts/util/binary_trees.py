import os
import glob
import xarray as xr
import rioxarray as rxr
import rasterio

# +
# Change directory to this repo - this should work on gadi or locally via python or jupyter.
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
print(src_dir)

from shelterbelts.apis.worldcover import tif_categorical, worldcover_labels
# -

cmap_woody_veg = {
    0: (240, 240, 240), # Non-trees are white
    1: (0, 100, 0),   # Trees are green
    255: (0, 100, 200)  # Nodata is blue
}


def worldcover_trees(filename, outdir):
    """Convert a worldcover tif into a binary tree cover tif"""
    da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

    # Trees or Shrubland
    da_trees = (da == 10) | (da == 20)
    da_trees = da_trees.astype('uint8')
        
    stub = filename.split('_')[-2]
    outpath = os.path.join(outdir, f"{stub}_woody_veg.tif")
    tif_categorical(da_trees, outpath, cmap_woody_veg, tiled=True)
    
    # Add pyramid for faster viewing in zoomed out views QGIS
    levels  = [2, 4, 8, 16, 32, 64]
    with rasterio.open(outpath, "r+") as src:          
        src.build_overviews(levels)
    


def run_tifs(folder, func, outdir):
    """Run the function on every tif file in a folder"""
    tif_files = glob.glob(os.path.join(folder, "*.tif*"))
    for i, tif_file in enumerate(tif_files):
        print(f"{i+1}/{len(tif_files)}: Working on {tif_file}")
        worldcover_trees(tif_file, outdir)


# %%time
if __name__ == '__main__':
    folder = '/scratch/xe2/cb8590/Worldcover_Australia'
    filename = os.path.join(folder,'ESA_WorldCover_10m_2021_v200_S36E147_Map.tif')
    outdir = '/scratch/xe2/cb8590/worldcover_trees'
    # worldcover_trees(filename, outdir)
    
    run_tifs(folder, worldcover_trees, outdir)


