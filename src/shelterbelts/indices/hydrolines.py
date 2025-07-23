# +
import os
import argparse
import geopandas as gpd
import rioxarray as rxr
from rasterio.features import rasterize

from shapely.geometry import box

from shelterbelts.apis.worldcover import tif_categorical
from shelterbelts.indices.catchments import gullies_cmap

# -

def hydrolines(geotif, hydrolines_gdb, outdir=".", stub="TEST"):
    """Crop the hydrolines to the region of interest
    
    Parameters
    ----------
        geotif: String of filename to be used for the bounding box to crop the hydrolines
        hydrolines_gdb: String of filename downloaded from here https://researchdata.edu.au/surface-hydrology-lines-regional/3409155
        rasterize: Whether to convert to a raster or not

    Returns
    -------
    ds: xarray.DataSet with 'terrain' and 'gullies' layers

    Downloads
    ---------
    hydrolines_cropped.gpkg: Cropped hydrolines for the region of interest
    hydrolines.tif: A georeferenced tif file of the gullies based on hydrolines

    """
    # Read raster to get bounding box and CRS
    da = rxr.open_rasterio(geotif, masked=True).isel(band=0)
    raster_bounds = da.rio.bounds()
    raster_crs = da.rio.crs

    # This may take a while since we're loading 2GB into memory
    gdf = gpd.read_file(hydrolines_gdb, layer='HydroLines') # 2GB
    hydrolines_crs = gdf.crs

    # Reproject raster bounding box to hydrolines CRS (more computationally efficient than the other way around)
    bbox_geom = box(*raster_bounds)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=raster_crs)
    bbox_gdf = bbox_gdf.to_crs(hydrolines_crs)

    gdf_cropped = gpd.clip(gdf, bbox_gdf) # 7 secs for 2kmx2km test region with 30 hydrolines
    cropped_path = os.path.join(outdir, f"{stub}_hydrolines_cropped.gpkg")
    gdf_cropped.to_file(cropped_path)
    print("Saved", cropped_path)

    gdf_cropped = gdf_cropped.to_crs(da.rio.crs)
    shapes = [(geom, 1) for geom in gdf_cropped.geometry]
    transform = da.rio.transform()
    hydro_gullies = rasterize(
        shapes,
        out_shape=da.shape,
        transform=transform,
        fill=0
    )
    ds = da.to_dataset(name='terrain')
    ds['gullies'] = (["y", "x"], hydro_gullies)

    filename_hydrolines = os.path.join(outdir, f"{stub}_hydrolines.tif")
    tif_categorical(ds['gullies'], filename_hydrolines, colormap=gullies_cmap)

    return ds


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

    gdf = hydrolines(geotif, hydrolines_gdb, outdir, stub)


# +
# hydrolines_gdb = "/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/SurfaceHydrologyLinesRegional.gdb"
# outdir = '../../../outdir/'
# stub = 'g2_26729'
# rasterize = True
# geotif = f"{outdir}{stub}_categorised.tif"
# ds = hydrolines(geotif, hydrolines_gdb)
# -
