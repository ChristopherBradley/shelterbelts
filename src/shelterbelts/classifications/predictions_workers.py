# +
import os
import argparse
import subprocess

import numpy as np
import geopandas as gpd


# +
# Only using this for the predictions_filename() right now, so it can run in a jupyter notebook when testing

# Change directory to this repo
import sys, os
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}/src")
elif os.path.basename(os.getcwd()) != repo_name:
    repo_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from repo root
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)
print(f"Running from {repo_dir}")
# -


predictions_filename = os.path.join(repo_dir, 'shelterbelts/classifications/predictions_batch.py')


def predictions_workers(gpkg, outdir, num_workers=10, year=2020, nn_dir='/g/data/xe2/cb8590/models', nn_stub='fft_89a_92s_85r_86p', limit=None):
    """Run predictions_batch in parallel with n workers at once
    
    Parameters  (same as predictions_batch + num_workers)
    ----------
        gpkg: Geopackage with the bounding box for each tile to download. A stub gets automatically assigned based on the center of the bbox.
        outdir: Folder to save the output tifs.
        num_workers: Integer number of processes to launch.
        year: The year of sentinel imagery to use as input for the tree predictions.
        nn_dir: The directory containing the neural network model and scaler.
        nn_stub: The stub of the neural network and preprocessing scaler model to make the predictions.
        limit: Number of rows for each worker to process. 'None' means process all the rows.
    
    Downloads
    ---------
        A tif with tree classifications for each bbox in the gpkg
    
    """
    
    gdf = gpd.read_file(gpkg)
    batch_dir = os.path.join(nn_dir, "batches")
    batch_stub = gpkg.split('/')[-1].split('.')[0]
    if not os.path.exists(batch_dir):
        os.mkdir(batch_dir)
    
    gdf_chunks = [gdf.iloc[i:i+num_workers] for i in range(0, len(gdf), num_workers)]
    
    procs = []
    for i, gdf_chunk in enumerate(gdf_chunks):
        if limit:
            gdf_chunk = gdf_chunk[:int(limit)]
        filename = os.path.join(batch_dir, f'{batch_stub}_num_workers{num_workers}_batch{i}.gpkg')
        gdf_chunk.to_file(filename)
        print(f"Created batch: {filename}, with num rows: {len(gdf_chunk)}", flush=True)

        p = subprocess.Popen(["python", predictions_filename, "--gpkg", filename, "--outdir", outdir, "--year", str(year), "--nn_dir", nn_dir, "--nn_stub", nn_stub])
        procs.append(p)
        print(f"Launched process: {i}", flush=True)

    for p in procs:
        p.wait()
    


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gpkg", type=str, required=True, help="filename containing the tiles to use for bounding boxes. Just uses the geometry, and assigns a stub based on the central point")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for the final classified tifs")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of processes to launch")
    parser.add_argument("--year", type=int, default=2020, help="Year of satellite imagery to download for doing the classification")
    parser.add_argument("--nn_dir", type=str, default='/g/data/xe2/cb8590/models', help="The stub of the neural network model and preprocessing scaler")
    parser.add_argument("--nn_stub", type=str, default='fft_89a_92s_85r_86p', help="The stub of the neural network model and preprocessing scaler")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows for each worker to process. 'None' means every row that worker is assigned")

    return parser.parse_args()



if __name__ == '__main__':

    args = parse_arguments()
    
    gpkg = args.gpkg
    outdir = args.outdir
    num_workers = int(args.num_workers)
    year = int(args.year)
    nn_dir = args.nn_dir
    nn_stub = args.nn_stub
    limit = args.limit
    
    predictions_workers(gpkg, outdir, num_workers, year, nn_dir, nn_stub, limit)


# +
# # %%time
# filename = '/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_10.gpkg'
# outdir = '/scratch/xe2/cb8590/tmp'
# num_workers = 10
# predictions_workers(filename, outdir, num_workers, limit=1)
# -


