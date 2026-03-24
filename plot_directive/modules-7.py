import rioxarray
from shelterbelts.apis.canopy_height import visualise_canopy_height
from shelterbelts.utils.filepaths import get_filename

chm_tif = get_filename('g2_26729_chm_res10_filled.tif')
da = rioxarray.open_rasterio(chm_tif)
ds = da.to_dataset(dim='band').rename({1: 'canopy_height'}).squeeze()
visualise_canopy_height(ds)