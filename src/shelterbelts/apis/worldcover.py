# +
# Example code for using the planetary computer worldcover API is here: 
# https://planetarycomputer.microsoft.com/dataset/esa-worldcover#Example-Notebook


# +
# %%time
import os
import argparse

import numpy as np
import rioxarray # Even though this variable isn't used directly, it's needed for the da.rio methods
from pyproj import Transformer

import odc.stac
import pystac_client
import planetary_computer


from shelterbelts.utils.visualisation import tif_categorical, visualise_categories
# -

worldcover_cmap = {
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (250, 0, 0),
    60: (180, 180, 180),
    70: (240, 240, 240),
    80: (0, 100, 200),
    90: (0, 150, 160),
    95: (0, 207, 117),
    100: (250, 230, 160)
}
worldcover_labels = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss and lichen'
}


def worldcover_bbox(bbox=[147.736, -42.912, 147.786, -42.862], crs="EPSG:4326"):
    """Download worldcover data for a specific bounding box"""
    
    # Convert to EPSG:4326 because this crs is needed for the catalog search
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    bbox_4326 = [minx, miny, maxx, maxy]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox_4326
    )
    items = list(search.items())
    items = [items[0]]
    ds = odc.stac.load(items, crs="EPSG:4326", bbox=bbox_4326)
    da = ds.isel(time=0)['map']
    return da


def worldcover_centrepoint(lat=-34.389, lon=148.469, buffer=0.05):
    """Download worldcover using a lat, lon & buffer"""
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    crs="EPSG:4326"
    da = worldcover_bbox(bbox, crs)
    return da


def worldcover(lat=-34.389, lon=148.469, buffer=0.01, outdir=".", stub="TEST", save_tif=True, plot=True):
    """
    Download ESA WorldCover imagery from the Microsoft Planetary Computer API.

    Parameters
    ----------
    lat : float, optional
        Latitude in WGS 84 (EPSG:4326). Default is -34.389.
    lon : float, optional
        Longitude in WGS 84 (EPSG:4326). Default is 148.469.
    buffer : float, optional
        Distance in degrees in a single direction (0.01 ≈ 1 km),
        resulting in an approximately square area of size 2*buffer.
        Default is 0.01.
    outdir : str, optional
        Output directory for saving results. Default is current directory.
    stub : str, optional
        Prefix for output filenames. Default is "TEST".
    save_tif : bool, optional
        Whether to save a GeoTIFF with embedded color map. Default is True.
    plot : bool, optional
        Whether to save a PNG visualisation (not geolocated). Default is True.

    Returns
    -------
    xarray.Dataset
        Dataset with variable **worldcover** (integer codes) and
        latitude/longitude coordinates. The mapping of codes to classes
        is provided in ``worldcover_labels``.

    Notes
    -----
    When ``save_tif=True``, it writes:
    ``{stub}_worldcover.tif``

    When ``plot=True``, it writes:
    ``{stub}_worldcover.png``

    Examples
    --------
    Download a small tile without saving files:

    >>> ds = worldcover(buffer=0.01, save_tif=False, plot=False)
    Starting worldcover.py
    >>> 'worldcover' in ds.data_vars
    True

    Visualising the WorldCover output:

    .. plot::

        from shelterbelts.apis.worldcover import worldcover, worldcover_cmap, worldcover_labels
        from shelterbelts.utils.visualisation import visualise_categories
        
        ds = worldcover(buffer=0.01, save_tif=False, plot=False)
        visualise_categories(ds['worldcover'], colormap=worldcover_cmap, labels=worldcover_labels, title="ESA WorldCover")

    """
    print("Starting worldcover.py")

    max_buffer = 0.2   # 0.5 had a bug with large portions of the returned tif being black
    if buffer > max_buffer:
        buffer = max_buffer
        print(f"Area too large, please download in smaller tiles. Reducing buffer to {max_buffer}.") 
        print(f"Estimated filesize = 10MB, estimated download time = 2 mins")
    da = worldcover_centrepoint(lat, lon, buffer)
    ds = da.to_dataset().drop_vars(['time']).rename({'map': 'worldcover'})

    if save_tif:
        filename = os.path.join(outdir, f"{stub}_worldcover.tif")    
        tif_categorical(da, filename, worldcover_cmap)

    if plot:
        filename = os.path.join(outdir, f"{stub}_worldcover.png")    
        visualise_categories(da, filename, worldcover_cmap, worldcover_labels, "ESA WorldCover")

    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lat', default=-34.389, type=float, help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default=148.469, type=float, help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default=0.01, type=float, help='Buffer in each direction in degrees (default: 0.01 ≈ 1 km)')
    parser.add_argument('--outdir', default='.', help='The directory to save the outputs. (Default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--no-save-tif', dest='save_tif', action="store_false", default=True, help='Disable saving GeoTIFF output (default: enabled)')
    parser.add_argument('--no-plot', dest='plot', action="store_false", default=True, help='Disable PNG visualisation (default: enabled)')

    return parser


# %%time
if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    
    worldcover(args.lat, args.lon, args.buffer, args.outdir, args.stub, save_tif=args.save_tif, plot=args.plot)