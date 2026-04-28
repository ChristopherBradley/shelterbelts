import os
import argparse

import osmnx as ox
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box
from rasterio.features import rasterize
from osmnx._errors import InsufficientResponseError

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

    try:
        roads = ox.features_from_bbox(bbox_list, {"highway": highway_types})
    except InsufficientResponseError:
        roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])] if len(roads) else roads

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
