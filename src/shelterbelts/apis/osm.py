import os
import argparse

import geopandas as gpd
import requests
import rioxarray as rxr
from shapely.geometry import box, LineString
from rasterio.features import rasterize

from shelterbelts.utils.visualisation import tif_categorical

roads_cmap = {
    0: (255, 255, 255),
    1: (138, 126, 125),
}
roads_labels = {
    0: "Non-roads",
    1: "Roads",
}
highway_types = ["motorway", "trunk", "primary", "secondary", "tertiary"]

# overpass-api.de round-robins between backends (gall and lambert), and either one
# can return 504 while the other is healthy - which one you get is down to DNS, so
# the same query fails on one machine and succeeds on another. Retrying re-resolves
# the name, and the mirrors are there for when the whole service is struggling.
#
# Every endpoint here must serve the full planet. A regional instance answers 200
# with zero ways outside its own country, which would silently look like "no roads"
# rather than an error - overpass.osm.ch does exactly that for Australia.
overpass_endpoints = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
overpass_passes = 2
overpass_timeout = 90
user_agent = "shelterbelts (https://github.com/ChristopherBradley/shelterbelts)"


def _overpass_ways(bbox_list):
    """Fetch the highway ways intersecting a bounding box, trying each endpoint in turn."""
    west, south, east, north = bbox_list
    query = (
        f'[out:json][timeout:60];'
        f'(way["highway"~"^({"|".join(highway_types)})$"]({south},{west},{north},{east}););'
        f'out geom;'
    )

    failures = []
    for _ in range(overpass_passes):
        for url in overpass_endpoints:
            try:
                response = requests.post(url, data=query, headers={"User-Agent": user_agent},
                                         timeout=overpass_timeout)
            except requests.RequestException as e:
                failures.append(f"{url}: {type(e).__name__}")
                continue
            if response.status_code != 200:
                failures.append(f"{url}: HTTP {response.status_code}")
                continue
            return response.json()["elements"]

    raise RuntimeError("No Overpass endpoint answered: " + ", ".join(failures))


def osm_roads(geotif_or_da, outdir=".", stub="TEST", savetif=True, save_gpkg=True):
    """Download roads from OpenStreetMap for the region of interest.

    Parameters
    ----------
    geotif_or_da : str or xarray.DataArray
        Path to a GeoTIFF, or a DataArray, used to define the bounding box.
    outdir : str, optional
        Output directory.
    stub : str, optional
        Prefix for output filenames.
    savetif : bool, optional
        Whether to save a roads GeoTIFF.
    save_gpkg : bool, optional
        Whether to save a roads GeoPackage.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Road features in the region of interest.
    ds : xarray.Dataset
        Dataset with a roads layer rasterised to the input grid.
    """
    if isinstance(geotif_or_da, str):
        da = rxr.open_rasterio(geotif_or_da, masked=True).isel(band=0)
    else:
        da = geotif_or_da

    bbox_geom = box(*da.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=da.rio.crs)
    bbox_gdf = bbox_gdf.to_crs("EPSG:4326")
    bbox_list = list(bbox_gdf.total_bounds)

    rows = []
    for way in _overpass_ways(bbox_list):
        points = way.get("geometry") or []
        if len(points) < 2:
            continue
        rows.append({
            **way.get("tags", {}),
            "osmid": way["id"],
            "geometry": LineString([(p["lon"], p["lat"]) for p in points]),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if save_gpkg:
        filename = os.path.join(outdir, f"{stub}_roads.gpkg")
        gdf.to_file(filename, layer="main_roads")
        print("Saved:", filename)

    gdf = gdf.to_crs(da.rio.crs)
    shapes = [(geom, 1) for geom in gdf.geometry]
    transform = da.rio.transform()
    roads_raster = rasterize(shapes, out_shape=da.shape, transform=transform, fill=0)

    ds = da.to_dataset(name='geotif')
    ds['roads'] = (["y", "x"], roads_raster)

    if savetif:
        filename = os.path.join(outdir, f"{stub}_roads.tif")
        tif_categorical(ds['roads'], filename, colormap=roads_cmap)

    return gdf, ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('geotif', help='GeoTIFF path used for the bounding box')
    parser.add_argument('--outdir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--stub', default='TEST', help='Prefix for output files (default: TEST)')
    parser.add_argument('--no-save-tif', dest='savetif', action='store_false', default=True, help='Disable saving GeoTIFF output (default: enabled)')
    parser.add_argument('--no-save-gpkg', dest='save_gpkg', action='store_false', default=True, help='Disable saving GeoPackage output (default: enabled)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    gdf, ds = osm_roads(args.geotif, outdir=args.outdir, stub=args.stub,
                        savetif=args.savetif, save_gpkg=args.save_gpkg)
