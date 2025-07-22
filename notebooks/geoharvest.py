import rioxarray as rxr
from shapely.geometry import box
import geopandas as gpd


from geodata_harvester import getdata_dem

outdir = '../outdir/'
stub = 'g2_26729'
geotif = f"{outdir}{stub}_categorised.tif"

da = rxr.open_rasterio(geotif, masked=True).isel(band=0)
bbox_geom = box(*da.rio.bounds())
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=da.rio.crs)
bbox = bbox_gdf.to_crs("EPSG:4326").total_bounds.tolist()

outpath = "geoharvest_dem.tif"

getdata_dem.getwcs_dem(bbox, outpath)

