# +
# How long does it take to read in the rasters directly instead of using the open data cube?
# -

import rioxarray as rxr
import xarray as xr
import os
import numpy as np
import rasterio as rio

filename = '/g/data/ka08/ga/ga_s2bm_ard_3/51/JYG/2021/06/11/20210611T041036/ga_s2bm_nbart_3-2-1_51JYG_2021-06-11_final_band04.tif'

# !ls /g/data/ka08/ga/ga_s2bm_ard_3/51/JYG/2021/06/11/20210611T041036/

# %%time
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')

# +
# Folder with the data
folder = "/g/data/ka08/ga/ga_s2bm_ard_3/51/JYG/2021/06/11/20210611T041036"

# Mapping of band file suffixes to DEA measurement names
band_mapping = {
    "band01": "nbart_coastal_aerosol",
    "band02": "nbart_blue",
    "band03": "nbart_green",
    "band04": "nbart_red",
    "band05": "nbart_red_edge_1",
    "band06": "nbart_red_edge_2",
    "band07": "nbart_red_edge_3",
    "band08": "nbart_nir_1",
    "band8A": "nbart_nir_2",
    "band11": "nbart_swir_2",
    "band12": "nbart_swir_3"
}

# Only keep the bands you're interested in
wanted_measurements = [
    'nbart_blue', 'nbart_green', 'nbart_red', 
    'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
    'nbart_nir_1', 'nbart_nir_2',
    'nbart_swir_2', 'nbart_swir_3'
]

# Reverse map to go from measurement name to band file suffix
inv_map = {v: k for k, v in band_mapping.items()}
# -

# %%time
# Load bands into a dict of DataArrays
band_dataarrays = {}
for meas in wanted_measurements:
    band_code = inv_map[meas]
    if band_code == "band8A":
        band_filename = f"ga_s2bm_nbart_3-2-1_51JYG_2021-06-11_final_band08a.tif"
    else:
        band_filename = f"ga_s2bm_nbart_3-2-1_51JYG_2021-06-11_final_{band_code}.tif"

    path = os.path.join(folder, band_filename)
    da = rxr.open_rasterio(path, masked=True).squeeze("band", drop=True)
    band_dataarrays[meas] = da

# +
# %%time
# Load the cloud mask: 1 = clear, 2 = cloudy, nan = No data
cloud_mask_path = os.path.join(folder, "ga_s2bm_oa_3-2-1_51JYG_2021-06-11_final_s2cloudless-mask.tif")
cloud_mask = rxr.open_rasterio(cloud_mask_path, masked=True).squeeze("band", drop=True)

bands_10m = [
    'nbart_blue', 'nbart_green', 'nbart_red', 
    'nbart_nir_1'
]

bands_20m = [
    'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
    'nbart_nir_2', 'nbart_swir_2', 'nbart_swir_3'
]

ref_10m = band_dataarrays['nbart_red']
cloud_mask_10m = cloud_mask.rio.reproject_match(ref_10m, resampling=rio.enums.Resampling.nearest)

ref_20m = band_dataarrays['nbart_swir_2']
cloud_mask_20m = cloud_mask.rio.reproject_match(ref_20m, resampling=rio.enums.Resampling.nearest)

# +
# %%time
# Apply the cloud mask
for key in bands_10m:
    band_dataarrays[key] = band_dataarrays[key].where(cloud_mask_10m == 1)
    
for key in bands_20m:
    band_dataarrays[key] = band_dataarrays[key].where(cloud_mask_20m == 1)
# -

# %%time
# Upsample 20 m bands to match 10 m grid using bilinear interpolation
for key in bands_20m:
    band_dataarrays[key] = band_dataarrays[key].rio.reproject_match(ref_10m, resampling=rio.enums.Resampling.nearest)

# +
# time = xr.DataArray(["2021-06-11"], dims=["time"])
# for key in band_dataarrays:
#     band_dataarrays[key] = band_dataarrays[key].expand_dims(time=time)

ds = xr.Dataset(band_dataarrays)
# -

ds
