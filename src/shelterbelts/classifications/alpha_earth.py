# +
import os
import glob
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

import geemap
import ee
# -
from shelterbelts.apis.worldcover import tif_categorical, worldcover_labels


# +
def download_alphaearth_da(da, start_date="2020-01-01", end_date="2021-01-01", outdir=".", stub="TEST", save=True, authenticate=False):
    """Download alphaearth embeddings for a DataSet and time period of interest."""
    if authenticate:
        ee.Authenticate()  # Apparently I only need to authenticate once in a notebook, then I can just use the initialise option when running pbs scripts
    
    ee.Initialize()
    
    # Prep the embeddings
    bbox = da.rio.bounds()
    polygon_coords = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])]
    roi = ee.Geometry.Polygon([polygon_coords]) # Should expand this by a few pixels so we don't get NaN values when reproject_matching to the sentinel imagery
    collection = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
    )
    image = ee.Image(collection.first())  # Just one image if we just get a single year
    
    # Download the embeddings
    print(f"Downloading embeddings for stub: {stub}")
    np_array = geemap.ee_to_numpy(image, region=roi, scale=10)  # Took 20 secs for a 1km x 1km region for 2020
    
    # Create an xarray to align with the tree tif
    ny, nx, n_bands = np_array.shape
    x_coords = np.linspace(bbox[0], bbox[2], nx)
    y_coords = np.linspace(bbox[3], bbox[1], ny)  # top → bottom
    ae_da = xr.DataArray(
        np_array, 
        dims=("y", "x", "band"), 
        coords={"x": x_coords, "y": y_coords, "band": np.arange(n_bands)}, 
        name="alpha_earth"
    )
    ae_da = ae_da.rio.write_crs("EPSG:4326")
    ae_da = ae_da.transpose("band", "y", "x")
    ae_da_match = ae_da.rio.reproject_match(da)
    
    # Just save the pickle file so I can combine it with my sentinel imagery
    filename = os.path.join(outdir, f'{stub}_alpha_earth_embeddings_{start_date[:4]}.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(ae_da_match, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", filename)
    
    return ae_da_match
    
#     # Prepare the flattened tree outputs
#     tree_array = da.values  # shape (y, x), 0/1 labels
#     lon = da['x'].values
#     lat = da['y'].values
#     xx, yy = np.meshgrid(lon, lat)
#     coords = np.column_stack([xx.ravel(), yy.ravel()])
#     tree_flat = tree_array.ravel()
    
#     # Combine into a dataframe
#     n_bands, height, width = ae_da_match.shape
#     inputs_flat = ae_da_match.values.reshape(n_bands, -1).T 
#     tree_flat = da.values.ravel() 
#     columns = [f"emb_{i}" for i in range(n_bands)]
#     df = pd.DataFrame(inputs_flat, columns=columns)
#     df['tree'] = tree_flat
    
#     # Save to file
#     os.makedirs(outdir, exist_ok=True)
#     filename = os.path.join(outdir, f'{stub}_alpha_earth_embeddings_{start_date[:4]}.csv')
#     df.to_csv(filename)
#     print("Saved:", filename)
# -



def download_alphaearth_tif(tif, start_date="2020-01-01", end_date="2021-01-01", outdir="/scratch/xe2/cb8590/tmp", stub=None, save=True, authenticate=False):
    """Download alphaearth embeddings for a tif and time period of interest."""
    da = rxr.open_rasterio(tif).isel(band=0).drop_vars('band')
    da_4326 = da.rio.reproject('EPSG:4326')
    if stub is None:
        stub = tif.split('/')[-1].split('.')[0]  # The base filename
    ds = download_alphaearth_da(da_4326, start_date, end_date, outdir, stub, save, authenticate)
    return ds


def download_alphaearth_folder(folder, start_date="2020-01-01", end_date="2021-01-01", outdir="/scratch/xe2/cb8590/tmp", stub=None, save=True, authenticate=False, suffix='.tiff', limit=None):
    """Download alphaearth embeddings for a folder of tif files.
    
    Parameters:
        folder: Folder containing the tif files that we want to download matching embeddings
        start_date: First date of imagery to download
        end_date: Last date of imagery to download
        outdir: Output folder for the pickle file
        stub: Prefix of the pickle file
        save: Whether to save to file
        authenticate: Whether to do the authentication (should only need to do this once for the first time)
        suffix: The end name of each tif file in the folder
        limit: The number of tif files to loop through

    Downloads:
        {outdir}/{stub}_alpha_earth_embeddings.csv files for each tif in the folder
    """
    if authenticate:
        ee.Authenticate()  # Haven't totally figured this out... I think I need to always include it, but it only makes me use the url the first time in the notebook?
    
    tifs = glob.glob(f'{folder}/*{suffix}')
    if limit is not None:
        tifs = tifs[:limit]
    for tif in tifs:
        download_alphaearth_tif(tif, start_date, end_date, outdir, stub, save, authenticate=False)


# +
import argparse

def parse_arguments():
    """Parse command line arguments for download_alphaearth_folder() with default values."""
    parser = argparse.ArgumentParser(description="Download alphaearth embeddings for a folder of tif files.")

    parser.add_argument("folder", help="Folder containing the tif files")
    parser.add_argument("--start_date", default="2020-01-01", help="First date of imagery to download (default: 2020-01-01)")
    parser.add_argument("--end_date", default="2021-01-01", help="Last date of imagery to download (default: 2021-01-01)")
    parser.add_argument("--outdir", default="/scratch/xe2/cb8590/tmp", help="Output folder for the pickle file (default: /scratch/xe2/cb8590/tmp)")
    parser.add_argument("--stub", default=None, help="Prefix of the pickle file (default: None)")
    parser.add_argument("--save", action="store_true", help="Save to file (default: False)")
    parser.add_argument("--authenticate", action="store_true", help="Authenticate for alphaearth (default: False)")
    parser.add_argument("--suffix", default=".tiff", help="Suffix of tif files in the folder (default: .tiff)")
    parser.add_argument("--limit", type=int, default=None, help="Number of tif files to loop through (default: None)")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    folder = args.folder
    start_date = args.start_date
    end_date = args.end_date
    outdir = args.outdir
    stub = args.stub
    save = args.save
    authenticate = args.authenticate
    suffix = args.suffix
    limit = args.limit

    download_alphaearth_folder(
        folder,
        start_date=start_date,
        end_date=end_date,
        outdir=outdir,
        stub=stub,
        save=save,
        authenticate=authenticate,
        suffix=suffix,
        limit=limit,
    )


# +
# # %%time
# folder = f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
# year = 2020
# start_date = f"{year}-01-01"
# end_date = f"{year}-12-31"
# download_alphaearth_folder(folder, limit=2, authenticate=True)
