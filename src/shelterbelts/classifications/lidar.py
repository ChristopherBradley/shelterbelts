# +
# # !conda install -c conda-forge pdal python-pdal rasterio
# -

import pdal, json

filename = '/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz'
outpath = f'tree_raster_10m.tif'

# filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Milgadara/Point Clouds/AHD/Young201702-PHO3-C0-AHD_6306194_55_0002_0002.laz'
filename = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/Milgadara/Point Clouds/AHD/Young201709-LID1-C3-AHD_6306194_55_0002_0002.laz'
stub = 'g2_26729'
counts_tif = f'{stub}_10m_counts.tif'
tree_tif = f'{stub}_10m_binary.tif'


# +
# %%time
# Count the number of points with classification 5 (> 2m) that lie in each 10m pixel
pipeline = {
    "pipeline": [
        filename,
        {"type": "filters.range", "limits": "Classification[5:5]"},
        {"type": "writers.gdal",
         "filename": counts_tif,
         "resolution": 10,
         "output_type": "count",
         "gdaldriver": "GTiff"},
    ]
}
p = pdal.Pipeline(json.dumps(pipeline))
p.execute()

# Convert point counts into a binary raster
counts = rioxarray.open_rasterio(outpath).isel(band=0).drop_vars('band')
tree = (counts > 0).astype('uint8')
tree.rio.to_raster(tree_tif)
print("Saved:", tree_tif)

# +
# %%time
# Doing my own classifications 
dem_tif = "dem.tif"
dsm_tif = "dsm.tif"
chm_tif = "chm.tif"
resolution = 1.0

# ---------------- DEM pipeline ----------------
dem_json = {
    "pipeline": [
        {"type": "readers.las", "filename": filename},
        {"type": "filters.smrf"},  # classify ground
        {"type": "filters.range", "limits": "Classification[2:2]"},  # keep only ground
        {"type": "writers.gdal",
         "filename": dem_tif,
         "resolution": resolution,
         "gdaldriver": "GTiff",
         "output_type": "min",
         "nodata": -9999}
    ]
}
p_dem = pdal.Pipeline(json.dumps(dem_json))
p_dem.execute()


# ---------------- DSM pipeline ----------------
dsm_json = {
    "pipeline": [
        {"type": "readers.las", "filename": filename},
        {"type": "filters.range", "limits": "ReturnNumber[1:1]"},  # first returns
        {"type": "writers.gdal",
         "filename": dsm_tif,
         "resolution": resolution,
         "gdaldriver": "GTiff",
         "output_type": "max",
         "nodata": -9999}
    ]
}
p_dsm = pdal.Pipeline(json.dumps(dsm_json))
p_dsm.execute()


# ---------------- CHM pipeline ----------------
chm_json = {
    "pipeline": [
        {"type": "readers.las", "filename": filename},
        {"type": "filters.smrf"},  # classify ground
        {"type": "filters.hag_nn"},  # compute HeightAboveGround
        {"type": "writers.gdal",
         "filename": chm_tif,
         "resolution": resolution,
         "gdaldriver": "GTiff",
         "dimension": "HeightAboveGround",
         "output_type": "max",
         "nodata": -9999}
    ]
}
p_chm = pdal.Pipeline(json.dumps(chm_json))
p_chm.execute()


# +
import rioxarray
import numpy as np
from rasterio.enums import Resampling

tree_height_thresh = 2
tree_tif = "tree_binary_10m.tif"

# Open CHM
chm = rioxarray.open_rasterio("chm.tif")

# Reproject to 10 m resolution, using max aggregation
chm_10m = chm.rio.reproject(
    chm.rio.crs,
    resolution=10,
    resampling=Resampling.max
)

# Apply threshold
tree = (chm_10m > tree_height_thresh).astype(np.uint8)

# Ensure nodata = 0
tree = tree.rio.write_nodata(0)

# Save to GeoTIFF
tree.rio.to_raster(tree_tif, compress="LZW")

# -

print("Wrote:", dem_tif, dsm_tif, chm_tif, tree_tif)
