import rioxarray as rxr
import matplotlib.pyplot as plt

filename = '/Users/christopherbradley/Downloads/5-Bands-FPLAN101-and-104-orthophoto.tif'

da = rxr.open_rasterio(filename)

ds = da.to_dataset('band')

ds = ds.astype(int) # Need to increase precision before calculating indices or else you get overflow issues

B1 = ds[1]  # Blue
B2 = ds[2]  # Green
B3 = ds[3]  # Red
B4 = ds[4]  # Red-edge
B5 = ds[5]  # NIR

ds['NDVI'] = (B5 - B3) / (B5 + B3)
ds['EVI'] = 2.5 * ((B5 - B3) / (B5 + 6 * B3 - 7.5 * B1 + 1))

ds['NDVI'].plot()

ds['EVI'].plot()

B5.plot()

B3.plot()

ds['NDVI'].rio.to_raster(
    '/Users/christopherbradley/Downloads/Kowen_NDVI.tif',
    compress="lzw"
)

ds['EVI'].rio.to_raster(
    '/Users/christopherbradley/Downloads/Kowen_EVI.tif',
    compress="lzw"
)
