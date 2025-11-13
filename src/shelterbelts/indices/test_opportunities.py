

from shelterbelts.indices.opportunities import opportunities

# percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2024_actnsw_4326/subfolders/lat_34_lon_140/34_13-141_90_y2024_predicted.tif' # Should be fine
percent_tif='/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/expanded/lat_32_lon_148/32_01-148_02_y2018_predicted_expanded20.tif'

tmpdir = '/scratch/xe2/cb8590/'
stub='TEST'
cover_threshold=50
width=3
ridges=False
num_catchments=10
min_branch_length=10
contour_spacing=2
min_contour_length=1000
equal_area=True
crop_pixels=20

# ds = opportunities(percent_tif, tmpdir, stub, tmpdir, cover_threshold, 
#                    contour_spacing=contour_spacing, min_contour_length=min_contour_length, equal_area=equal_area,
#                   ridges=False, plot=True) # less than 1 second yay

ds = opportunities(percent_tif, tmpdir, stub, tmpdir, ridges=False, cover_threshold=50, crop_pixels=20, plot=True) # less than 1 second yay

# +
ds = opportunities(percent_tif, tmpdir, stub, tmpdir, ridges=False, equal_area=True, contour_spacing=0, cover_threshold=50, crop_pixels=20, plot=True) # less than 1 second yay


# -


