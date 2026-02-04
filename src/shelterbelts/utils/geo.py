"""Geographic and spatial utilities for shelterbelts analysis."""
# Note: I haven't yet tested this or de-duplicated from the original caopy_height file

import os
import shutil
import requests
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box


def transform_bbox(
    bbox=[148.464499, -34.394042, 148.474499, -34.384042],
    inputEPSG="EPSG:4326",
    outputEPSG="EPSG:3857"
):
    """Transform a bounding box from one coordinate system to another.
    
    Parameters
    ----------
    bbox : list
        Bounding box as [minx, miny, maxx, maxy]
    inputEPSG : str
        Input EPSG code
    outputEPSG : str
        Output EPSG code
    
    Returns
    -------
    tuple
        Transformed bbox as (minx, miny, maxx, maxy)
    """
    transformer = Transformer.from_crs(inputEPSG, outputEPSG, always_xy=True)
    # bbox = (minx, miny, maxx, maxy)
    x1, y1 = transformer.transform(bbox[0], bbox[1])
    x2, y2 = transformer.transform(bbox[2], bbox[3])
    return (x1, y1, x2, y2)


def identify_relevant_tiles_bbox(bbox=[147.735717, -42.912122, 147.785717, -42.862122], canopy_height_dir=".", footprints_geojson='tiles_global.geojson', id_column='tile'):
    """Find the tiles that overlap with the region of interest.
    
    Parameters
    ----------
    bbox : list
        Bounding box as [minx, miny, maxx, maxy]
    canopy_height_dir : str
        Directory containing or to store the footprints GeoJSON
    footprints_geojson : str
        Filename of the footprints GeoJSON
    id_column : str
        Column name containing tile IDs
    
    Returns
    -------
    list
        List of relevant tile IDs
    """
    roi_polygon = box(*bbox)
    
    # Download the 'tiles_global.geojson' to this folder if we haven't already
    filename = os.path.join(canopy_height_dir, footprints_geojson)
        
    if not os.path.exists(filename):
        assert footprints_geojson == 'tiles_global.geojson', 'Please provide footprints for the tifs in this directory'
        url = "https://s3.amazonaws.com/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/tiles.geojson"
        with requests.get(url, stream=True) as stream:
            with open(filename, "wb") as file:
                shutil.copyfileobj(stream.raw, file)
        print(f"Downloaded {filename}", flush=True)

    # Load the canopy height tiles
    gdf = gpd.read_file(filename)

    # Find any tiles that intersect with this polygon
    relevant_tiles = []
    for idx, row in gdf.iterrows():
        tile_polygon = row['geometry']
        if tile_polygon.intersects(roi_polygon):  
            relevant_tiles.append(row[id_column])
            
    return relevant_tiles
