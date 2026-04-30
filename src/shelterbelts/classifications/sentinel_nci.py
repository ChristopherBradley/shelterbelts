# This is mostly copied from the DEA example notebook: https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Sentinel2_Surface_Reflectance/#List-measurements

import argparse
import contextlib
import io
import os
import pickle
import sys
import traceback
import warnings

import geopandas as gpd
import rioxarray as rxr

# datacube and dea_tools are only installable inside the DEA/NCI sandbox. 
# Import lazily so the module can still be imported (and doctested) in regular environments.
try:
    import datacube
    from dea_tools.datahandling import load_ard
except ImportError:
    datacube = None
    load_ard = None

warnings.filterwarnings('ignore')


# Specific lat and lon range (instead of just lat, lon, buffer) so I can match the tiff files exactly
def define_query_range(lat_range, lon_range, time_range, input_crs='epsg:4326', output_crs='epsg:6933'):
    query = {
        'y': lat_range,
        'x': lon_range,
        'time': time_range,
        'resolution': (-10, 10),
        'crs': input_crs,
        'output_crs': output_crs,
        'group_by': 'solar_day',
    }
    return query


def load_and_process_data(dc, query):
    """Download all 10 bands from the DEA Datacube"""
    with contextlib.redirect_stdout(io.StringIO()): # Silence the print statements
        ds = load_ard(
            dc=dc,
            products=['ga_s2am_ard_3', 'ga_s2bm_ard_3', 'ga_s2cm_ard_3'],
            cloud_mask='s2cloudless',
            min_gooddata=0.9,
            measurements=['nbart_blue', 'nbart_green', 'nbart_red',
                          'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
                          'nbart_nir_1', 'nbart_nir_2',
                          'nbart_swir_2', 'nbart_swir_3'],
            **query,
        )
    return ds


def download_ds2(tif, start_date="2020-01-01", end_date="2021-01-01", outdir=".", save=True):
    """Download Sentinel imagery matching the bounding box of an existing tif file."""
    da = rxr.open_rasterio(tif).isel(band=0).drop_vars('band')
    da_4326 = da.rio.reproject('EPSG:4326')
    bbox = da_4326.rio.bounds()
    stub = tif.split('/')[-1].split('.')[0]
    return download_ds2_bbox(bbox, start_date, end_date, outdir, stub, save)


def download_ds2_bbox(bbox, start_date="2020-01-01", end_date="2021-01-01", outdir=".", stub="TEST", save=True, input_crs='epsg:4326'):
    """
    Download a Sentinel-2 surface reflectance stack for a bounding box and date range. Uses a 10% cloud threshold and downloads all 10 bands.

    .. important::
       This function only runs inside the DEA/NCI datacube environment.

    Parameters
    ----------
    bbox : tuple of float
        (minx, miny, maxx, maxy) in input_crs.
    start_date, end_date : str, optional
        First and last date of imagery to download.
    outdir : str, optional
        Output directory for saving results.
    stub : str, optional
        Prefix for the output filename.
    save : bool, optional
        Save the Dataset to {outdir}/{stub}_ds2_{year}.pkl.
    input_crs : str, optional
        CRS of the bbox.

    Returns
    -------
    xarray.Dataset
        Ten-band Sentinel-2 stack with dims (time, y, x) in EPSG:3857.

    Notes
    -----
    EPSG:3857 is used as the output CRS because EPSG:28355 / EPSG:3577 was roughly 50% slower in my tests.
    """
    lat_range = (bbox[1], bbox[3])
    lon_range = (bbox[0], bbox[2])
    time_range = (start_date, end_date)
    output_crs = 'EPSG:3857'
    query = define_query_range(lat_range, lon_range, time_range, input_crs, output_crs)

    dc = datacube.Datacube(app='sentinel_download')
    ds = load_and_process_data(dc, query)

    if save:
        year = start_date[:4]
        filename = os.path.join(outdir, f'{stub}_ds2_{year}.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {filename}", flush=True)

    return ds


def run_download_gdf(gdf, outdir, start_date='2020-01-01', end_date='2021-01-01'):
    """Run :func:`download_ds2_bbox` on every row in the GeoDataFrame."""
    for _, row in gdf.iterrows():
        filename = row['filename']
        if 'start_date' in gdf.columns and 'end_date' in gdf.columns:
            start_date = row['start_date']
            end_date = row['end_date']
        stub = filename.split('.')[0] + '_' + start_date[:4]
        bounds = row['geometry'].bounds

        try:
            print(f"Downloading: {stub}_{start_date}_{end_date}", flush=True)
            download_ds2_bbox(bounds, start_date, end_date, outdir, stub)
        except Exception:
            print(f"Error in downloading: {stub}_{start_date}_{end_date}:", flush=True)
            traceback.print_exc(file=sys.stdout)


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("gpkg", type=str, help="A gpkg with: (filename, [start_date], [end_date], geometry), generated by bounding_boxes.py. The start_date and end_date columns override the CLI arguments.")
    parser.add_argument("outdir", type=str, help="Output directory for the pickle files")
    parser.add_argument("--limit", type=int, default=None, help="Number of files to process")
    parser.add_argument("--start_date", type=str, default='2020-01-01', help="Starting date to download sentinel imagery (default='2020-01-01')")
    parser.add_argument("--end_date", type=str, default='2021-01-01', help="Ending date to download sentinel imagery (default='2021-01-01')")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    gdf = gpd.read_file(args.gpkg)
    if args.limit is not None:
        gdf = gdf[:args.limit]
    run_download_gdf(gdf, args.outdir, args.start_date, args.end_date)
