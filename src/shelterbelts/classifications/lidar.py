# Using the pre-classified values to create a 10m tree raster
import pdal, json
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



# +
# # !pip install laspy[lazrs,laszip]
# -

import laspy
import numpy as np
import rioxarray as rxr
import xarray as xr

filename = '/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz'


las = laspy.read(filename)
z = las.z 


tree_code = 5  # Classification code for trees > 2m
tree_points = np.isin(las.classification, [tree_code])

da = xr.DataArray(H, dims=("y", "x"), coords={"y": y_coords[::-1], "x": x_coords}, name="tree")
da.rio.write_crs("EPSG:4326", inplace=True)
da.rio.to_raster("tree_raster10.tif")

# +
# Did a conda install of pdal
# -





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
# -







