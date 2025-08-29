import rioxarray as rxr
import matplotlib.pyplot as plt

stub = 'milagadara'
filename = '/Users/christopherbradley/Documents/PHD/Data/Drone/Milgadara_2025-07-16/Barwang-16-07-2025-all/odm_orthophoto/odm_orthophoto.tif'

da = rxr.open_rasterio(filename)

da

ds = da.to_dataset('band')

ds = ds.astype(int) # Need to increase precision before calculating indices or else you get overflow issues

B1 = ds[1]  # Blue
B2 = ds[2]  # Green
B3 = ds[3]  # Red
B4 = ds[4]  # Red-edge
B5 = ds[5]  # NIR

ds['NDVI'] = (B5 - B3) / (B5 + B3)

ds['NDVI'].rio.to_raster(
    f'/Users/christopherbradley/Downloads/{stub}_NDVI.tif',
    compress="lzw"
)

ds['NDVI'].plot()

B5.plot()

B3.plot()


