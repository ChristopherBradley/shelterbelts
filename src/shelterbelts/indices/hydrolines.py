# Crop the hydrolines to the region of interest

import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box
import pyproj

hydrolines_gdb = "/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/SurfaceHydrologyLinesRegional.gdb"

# %%time
gdf = gpd.read_file(hydrolines_gdb, layer='HydroLines') # 2GB


def hydrolines(geotif, hydrolines_gdb, outdir=".", stub="TEST"):
    """Crop the hydrolines to the region of interest
    
    Parameters
    ----------
        geotif: File to be used for the bounding box to crop the hydrolines
        hydrolines_gdb: Downloaded from here https://researchdata.edu.au/surface-hydrology-lines-regional/3409155

    Returns
    -------
        gdf: hydrolines cropped to the region of interest

    Downloads
    ---------
        hydrolines_cropped.gpkg: A geopackage of the hydrolines for the region of interest
    """


outdir = '../../../outdir/'
stub = 'g2_26729'
geotif = f"{outdir}{stub}_categorised.tif"

# +
# Read raster to get bounding box and CRS
da = rxr.open_rasterio(geotif, masked=True).isel(band=0)
raster_bounds = da.rio.bounds()
raster_crs = da.rio.crs

hydrolines_crs = gdf.crs
# -

# Reproject raster bounding box to hydrolines CRS (more computationally efficient than the other way around)
bbox_geom = box(*raster_bounds)
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=raster_crs)
bbox_gdf = bbox_gdf.to_crs(hydrolines_crs)

# %%time
gdf_cropped = gpd.clip(gdf, bbox_gdf) # 7 secs

cropped_path = f"{outdir}{stub}_hydrolines_cropped.gpkg"
gdf_cropped.to_file(cropped_path)

print(cropped_path)

# !ls
