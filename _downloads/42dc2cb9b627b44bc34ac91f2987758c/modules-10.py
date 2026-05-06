from shelterbelts.indices.catchments import catchments
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import plot_catchments_sidebyside

dem_file = get_filename('g2_26729_DEM-H.tif')

ds5 = catchments(dem_file, stub='num_catchments5', num_catchments=5, savetif=False, plot=False)
ds20 = catchments(dem_file, stub='num_catchments20', num_catchments=20, savetif=False, plot=False)
plot_catchments_sidebyside(ds5, ds20, title1='num_catchments=5', title2='num_catchments=20')