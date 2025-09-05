# +
# Copying from PaddockTSLocal since the environment wasn't working on my machine
# https://github.com/thestochasticman/paddock-ts-local/blob/main/PaddockTS/Data/download_ds2.py

# There's also a nice example on the DEA website:
# https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Sentinel2_Surface_Reflectance/

# +
# # !pip install "dask[distributed]" cql2

# +
import pickle
import os

from xarray.core.dataset import Dataset
import rioxarray as rxr
import odc.stac
import pystac_client
from dask.distributed import Client as DaskClient


# I haven't bothered to create a function for downloading sentinel imagery from a folder of tifs with DEA like in sentinel_nci, because it takes sooo much longer than just doing this on NCI

def download_ds2(tif, start_date="2020-01-01", end_date="2021-01-01", outdir=".") -> Dataset:
    """Download sentinel imagery matching the bounding box of the tif file"""
    da = rxr.open_rasterio(tif).isel(band=0).drop_vars('band')
    da_4326 = da.rio.reproject('EPSG:4326')
    bbox = da_4326.rio.bounds()
    stub = tif.split('/')[-1].split('.')[0]
    ds2 = download_ds2_bbox(bbox, start_date, end_date, outdir, stub)
    return ds2

# Takes about 5 mins to download 1 year of data, compared to 40 secs using the datacube on gadi
def download_ds2_bbox(bbox, start_date="2020-01-01", end_date="2021-01-01", outdir=".", stub="TEST") -> Dataset:
    """
    Download sentinel imagery for the bounding box and time period of interest.

    Parameters:
        bbox: bounding box of the region of interest
        start_date: First date of imagery to download
        end_date: Last date of imagery to download
        outdir: Output folder for the pickle file
        stub: Prefix of the pickle file

    Returns:
        Dataset: The loaded xarray Dataset (also saved to `{outdir}/{stub}.pkl`).
    """
    catalog = pystac_client.Client.open('https://explorer.dea.ga.gov.au/stac')
    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={'aws_unsigned': True},
    )
    query_results = catalog.search(
        bbox=bbox,
        collections=['ga_s2am_ard_3', 'ga_s2bm_ard_3', 'ga_s2cm_ard_3'],
        datetime=f"{start_date}/{end_date}",
        filter="eo:cloud_cover < 10"
    )
    items = list(query_results.items())
    
    ds2 = odc.stac.load(
        items,
        bands=['nbart_blue', 'nbart_green', 'nbart_red', 
              'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
              'nbart_nir_1', 'nbart_nir_2',
              'nbart_swir_2', 'nbart_swir_3'],
        crs='utm',
        resolution=10,
        groupby='solar_day',
        bbox=bbox,
        # chunks={  # Lazy loading. Seems to break my merge_inputs_outputs currently.
        #     'time': 1,
        #     'x': 1024,
        #     'y': 1024
        # }
    )
    filename = os.path.join(outdir, f'{stub}_ds2_{start_date}.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(ds2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved", filename)

    ## Later you can load this pickle file using this. It seems to have the same filesize and efficiency and lazy loading as netcdf files from my experimenting.
    # with open('ds2.pkl', 'rb') as file:
    #     ds = pickle.load(file)
    
    return ds2



# +
import argparse

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tif', required=True, help='Filename of the tif to match the bounding box')
    parser.add_argument('--start_date', default='2020-01-01', help='First date of imagery to download (default: 2020-01-01)')
    parser.add_argument('--end_date', default='2021-01-01', help='Last date of imagery to download (default: 2021-01-01)')
    parser.add_argument('--outdir', default='.', help='Output folder for the pickle file (default: current directory)')
    parser.add_argument('--stub', default='TEST', help='Prefix of the pickle file (default: TEST)')

    return parser.parse_args()


# # %%time
if __name__ == '__main__':
    args = parse_arguments()
    
    download_ds2_bbox(
        bbox=args.bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        outdir=args.outdir,
        stub=args.stub
    )


# +
# # %%time
# tif = '../../../data/g2_26729_binary_tree_cover_10m.tiff'
# ds = download_ds2(tif)
# # ds['nbart_red'].isel(time=0).plot()
# stub
# # !ls