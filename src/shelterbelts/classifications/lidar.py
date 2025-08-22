# +
# Did a conda install of pdal
# -

import pdal, json

# +
filename = '/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz'


# -

# Using the pre-classified values to create a 10m tree raster
pipeline = {
    "pipeline": [
        filename,
        {"type": "filters.range", "limits": "Classification[5:5]"},
        {"type": "writers.gdal",
         "filename": "tree_raster10.tif",
         "resolution": 10,
         "output_type": "count",
         "gdaldriver": "GTiff"}
    ]
}
p = pdal.Pipeline(json.dumps(pipeline))
p.execute()




# I also tried out laspy, but had issues converting from the numpy array to a rioxarray or tif because I had to manually create the latitude and longitude coordinates
# laspy doesn't seem as powerful as pdal anyway (in terms of tree segmentation etc.)
las = laspy.read(filename)
z = las.z 
tree_code = 5  # Classification code for trees > 2m
tree_points = np.isin(las.classification, [tree_code])





# +
# ground classification & height normalization
# %%time
pipeline_json = {
    "pipeline": [
        filename,
        {"type": "filters.smrf"},
        {"type": "filters.hag_nn"},
        "out_normalized.laz"
    ]
}
pipeline = pdal.Pipeline(json.dumps(pipeline_json))
pipeline.execute()

# Took 26 secs
# +
# This just created the dem.tif, but the rest of the pipeline failed

# %%time
import json
import pdal
import rasterio
import numpy as np

filename = "/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz"

# Outputs (GeoTIFFs)
dem_tif = "dem.tif"       # ground surface
dsm_tif = "dsm.tif"       # canopy/first-return surface
chm_tif = "chm.tif"       # canopy height model (DSM - DEM)
tree_tif = "tree_binary.tif"
resolution = 1.0          # metres per pixel
tree_height_thresh = 2.0  # metres (tweak as needed)

# PDAL pipeline:
# - SMRF: classify ground
# - HAG:  HeightAboveGround
# - DEM:  min(Z) of ground-only points
# - DSM:  max(Z) of first returns
# - CHM:  max(HeightAboveGround)
pipeline_json = {
    "pipeline": [
        {"type": "readers.las", "filename": filename, "tag": "in"},

        {"type": "filters.smrf", "inputs": "in", "tag": "smrf"},
        {"type": "filters.hag_nn", "inputs": "smrf", "tag": "hag"},

        # DEM from ground (Class 2)
        {"type": "filters.range", "inputs": "smrf", "limits": "Classification[2:2]", "tag": "ground"},
        {"type": "writers.gdal",
         "inputs": "ground",
         "filename": dem_tif,
         "resolution": resolution,
         "gdaldriver": "GTiff",
         "output_type": "min",
         "nodata": -9999},

        # DSM from first returns
        {"type": "filters.range", "inputs": "in", "limits": "ReturnNumber[1:1]", "tag": "first"},
        {"type": "writers.gdal",
         "inputs": "first",
         "filename": dsm_tif,
         "resolution": resolution,
         "gdaldriver": "GTiff",
         "output_type": "max",
         "nodata": -9999},

        # CHM = max(HeightAboveGround)
        {"type": "writers.gdal",
         "inputs": "hag",
         "filename": chm_tif,
         "resolution": resolution,
         "gdaldriver": "GTiff",
         "dimension": "HeightAboveGround",
         "output_type": "max",
         "nodata": -9999}
    ]
}

p = pdal.Pipeline(json.dumps(pipeline_json))
p.execute()

# Binary tree raster from CHM (> threshold)
with rasterio.open(chm_tif) as src:
    chm = src.read(1)
    prof = src.profile.copy()

tree = (chm > tree_height_thresh).astype(np.uint8)
prof.update(dtype=rasterio.uint8, nodata=0)

with rasterio.open(tree_tif, "w", **prof) as dst:
    dst.write(tree, 1)

print("Wrote:", dem_tif, dsm_tif, chm_tif, tree_tif)


# +
# %%time
import json
import pdal

filename = "/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz"

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

# -




