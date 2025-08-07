# +
# # !pip install osmnx

# +
import os
import argparse

import osmnx as ox
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box
from rasterio.features import rasterize
from osmnx._errors import InsufficientResponseError 

from shelterbelts.apis.worldcover import tif_categorical

# -

roads_cmap = {
    0: (255, 255, 255),
    1: (138, 126, 125),
}
roads_labels = {
    0: "Non-roads",
    1: "Roads",
}
highway_types = ["motorway", "trunk", "primary", "secondary", "tertiary"]


def osm_roads(geotif, outdir=".", stub="TEST"):
    """Download roads from openstreetmaps for the region of interest
    
    Parameters
    ----------
        geotif: String of filename to be used for the bounding box to crop the hydrolines

    Returns
    -------
        gdf: geodataframe of the features in the region of interest
        ds: xarray.DataSet with a layer 'roads'

    Downloads
    ---------
        roads.gpkg: roads downloaded for the region of interest
        roads.tif: A georeferenced tif file of the roads in this region
        
    """
    # Get the bbox (could make this a function since I've used in hydrolines.py too)
    da = rxr.open_rasterio(geotif, masked=True).isel(band=0)
    bbox_geom = box(*da.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=da.rio.crs)
    bbox_gdf = bbox_gdf.to_crs("EPSG:4326")
    bbox_list = list(bbox_gdf.total_bounds)

    try:
        # Download the roads from Open Street Maps
        roads = ox.features_from_bbox(bbox_list, {"highway":highway_types})  # This took about 10 secs for my test 2km x 2km region with 1 main road
    except InsufficientResponseError:
        # Return an empty GeoDataFrame if there are no roads in this location
        roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Save as a geopackage
    gdf = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    filename = os.path.join(outdir, f"{stub}_roads.gpkg")
    gdf.to_file(filename, layer="main_roads")
    print("Saved:", filename)

    # Create a raster (Should also make this a function if I do something similar again)
    gdf = gdf.to_crs(da.rio.crs)
    shapes = [(geom, 1) for geom in gdf.geometry]
    transform = da.rio.transform()
    roads_raster = rasterize(
        shapes,
        out_shape=da.shape,
        transform=transform,
        fill=0
    )
    ds = da.to_dataset(name='geotif')
    ds['roads'] = (["y", "x"], roads_raster)
    filename = os.path.join(outdir, f"{stub}_roads.tif")
    tif_categorical(ds['roads'], filename, colormap=roads_cmap)

    return gdf, ds


# +
def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--geotif', help='String of filename to be used for the bounding box of the region of interest')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', default='TEST', help='Prefix for output files.')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arguments()

    geotif = args.geotif
    outdir = args.outdir
    stub = args.stub

    gdf, ds = osm_roads(geotif, outdir, stub)

# -

# outdir = '../../../outdir/'
# stub = 'g2_26729'
# geotif = f"{outdir}{stub}_categorised.tif"
# gdf, ds = osm_roads(geotif, outdir, stub)
