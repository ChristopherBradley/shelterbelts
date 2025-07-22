import argparse
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box


def hydrolines(geotif, hydrolines_gdb, outdir=".", stub="TEST"):
    """Crop the hydrolines to the region of interest
    
    Parameters
    ----------
        geotif: String of filename to be used for the bounding box to crop the hydrolines
        hydrolines_gdb: String of filename downloaded from here https://researchdata.edu.au/surface-hydrology-lines-regional/3409155

    Returns
    -------
        gdf: hydrolines cropped to the region of interest

    Downloads
    ---------
        hydrolines_cropped.gpkg: A geopackage of the hydrolines for the region of interest
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

    cropped_path = f"{outdir}{stub}_hydrolines_cropped.gpkg"
    gdf_cropped.to_file(cropped_path)

    return gdf_cropped


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
# geotif = f"{outdir}{stub}_categorised.tif"

## Code for rasterizing the hydrolines
# filename_hydrolines = os.path.join(outdir, f"{stub}_hydrolines_cropped.gpkg")
# gdf_hydrolines = gpd.read_file(filename_hydrolines)
# gdf_hydrolines_reprojected = gdf_hydrolines.to_crs(grid.crs)
# shapes = [(geom, 1) for geom in gdf_hydrolines_reprojected.geometry]
# hydro_gullies = rasterize(
#     shapes,
#     out_shape=acc.shape,
#     transform=grid.affine, 
#     fill=0
# )