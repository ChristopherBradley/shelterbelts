import time

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

# I haven't yet setup environments using pdal or lidr for creating tifs from .laz files on NCI
from shelterbelts.classifications.sentinel_nci import download_ds2
from shelterbelts.classifications.merge_inputs_outputs import tile_csv


# +
# Downloading sentinel imagery from the NCI Datacube

data_dir = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
stub = 'g2_26729_binary_tree_cover_10m'
sample_tif = f'{data_dir}/{stub}.tiff'
outdir = '/scratch/xe2/cb8590/tmp'

def test_basic():
    """Basic tests for each of the files"""
    download_ds2(sample_tif, start_date="2020-01-01", end_date="2021-01-01", outdir=outdir)


def test_sentinel():
    """More comprehensive tests for sentinel imagery from the DEA STAC API: 2x year ranges"""
    download_ds2(sample_tif, start_date="2020-06-01", end_date="2021-06-01", outdir=outdir)
    # 10 secs using the NCI datacube compared to 5 mins using the DEA STAC API


# -

def test_merging():
    """More comprehensive tests for the merging: 2x radius, 2x spacing"""
    sentinel_tile =  f'{outdir}/{stub}_ds2_2020.pkl'
    tile_csv(sentinel_tile, tree_file=sample_tif, outdir=outdir, radius=4, spacing=10, double_f=True)
    tile_csv(sentinel_tile, tree_file=sample_tif, outdir=outdir, radius=1, spacing=10, double_f=True)
    tile_csv(sentinel_tile, tree_file=sample_tif, outdir=outdir, radius=4, spacing=1, double_f=True)
    

# +

# %%
# Training a neural network


# %%
# Predicting new locations
# -

if __name__ == '__main__':
    print("testing classifications on NCI")
    start = time.time()
    
    # test_sentinel()
    test_merging()
    
    print(f"tests successfully completed in {time.time() - start} seconds")
