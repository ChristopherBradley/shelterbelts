# +
import os
import argparse

import numpy as np
import geopandas as gpd
import rioxarray as rxr
from rasterio.features import rasterize

from shapely.geometry import box

from shelterbelts.utils import tif_categorical
# from shelterbelts.apis.catchments import gullies_cmap  # This won't work with the DEA environment, since it needs DAESIM_preprocess to be pip installed

gullies_cmap = {
    0: (255, 255, 255),
    1: (0, 0, 255),
}
# -

def crop_and_rasterize(geotif, feature_gdb, outdir=".", stub="TEST", save_gpkg=True, savetif=True, layer='HydroLines', feature_name=None):
    """Crop vector features to the region of interest and rasterize them.
    
    This function generalizes rasterization of any vector features (hydrolines, roads, etc.)
    to the bounding box of a reference raster.
    
    Parameters
    ----------
        geotif: String of filename to be used for the bounding box to crop the features, or xarray DataArray
        feature_gdb: Path to GDB or GPKG file containing the vector features
        outdir: Output directory to save results
        stub: Prefix for output files
        save_gpkg: Whether to save cropped features as GPKG
        savetif: Whether to save rasterized output as GeoTIFF
        layer: Layer name within the GDB (e.g., 'HydroLines', 'NationalRoads_2025_09')
        feature_name: Name for the rasterized feature (e.g., 'gullies', 'roads'). 
                     Defaults to layer name converted to lowercase

    Returns
    -------
    gdf: geodataframe of the features in the region of interest
    ds: xarray.DataSet with rasterized feature layer

    """
    if feature_name is None:
        feature_name = layer.lower()
    
    # Use raster to get bounding box and CRS
    if isinstance(geotif, str):
        da = rxr.open_rasterio(geotif, masked=True).isel(band=0)
    else:
        da = geotif
    raster_bounds = da.rio.bounds()
    raster_crs = da.rio.crs

    # Reproject raster bounding box to feature CRS (more computationally efficient than the other way around)
    bbox_geom = box(*raster_bounds)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=raster_crs)
    bbox_gdf = bbox_gdf.to_crs('EPSG:4283')  # Standard GDB CRS

    if feature_gdb.endswith('.gpkg'):
        gdf = gpd.read_file(feature_gdb)  # pre-cropped geopackage
    else: 
        # This file is about 2GB, but can be spatially indexed so loads really fast
        gdf = gpd.read_file(feature_gdb, layer=layer, bbox=bbox_gdf)

    if save_gpkg:
        cropped_path = os.path.join(outdir, f"{stub}_{layer}_cropped.gpkg")
        gdf.to_file(cropped_path)
        print("Saved:", cropped_path)

    gdf = gdf.to_crs(da.rio.crs)
    shapes = [(geom, 1) for geom in gdf.geometry]
    transform = da.rio.transform()
    if not shapes:
        rasterized_feature = np.zeros(da.shape, dtype=np.uint8)
    else:
        rasterized_feature = rasterize(
            shapes,
            out_shape=da.shape,
            transform=transform,
            fill=0
        )
    ds = da.to_dataset(name='input')
    ds[feature_name] = (["y", "x"], rasterized_feature)

    if savetif:
        filename_feature = os.path.join(outdir, f"{stub}_{layer}.tif")
        tif_categorical(ds[feature_name], filename_feature, colormap=gullies_cmap)

    return gdf, ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--geotif', help='String of filename to be used for the bounding box to crop the hydrolines')
    parser.add_argument('--hydrolines_gdb', help='String of filename downloaded from here https://researchdata.edu.au/surface-hydrology-lines-regional/3409155')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', default='TEST', help='Prefix for output files.')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arguments()

    geotif = args.geotif
    hydrolines_gdb = args.hydrolines_gdb
    outdir = args.outdir
    stub = args.stub

    gdf, ds = crop_and_rasterize(geotif, hydrolines_gdb, outdir, stub, layer='HydroLines', feature_name='gullies')


# +
# # %%time
# hydrolines_gdb = "/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/SurfaceHydrologyLinesRegional.gdb"
# outdir = '../../../outdir/'
# stub = 'g2_26729'
# geotif = f"{outdir}{stub}_tree_categories.tif"
# gdf, ds = hydrolines(geotif, hydrolines_gdb)
# ds['gullies'].plot()
