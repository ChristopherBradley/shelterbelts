from shelterbelts.indices.shelter_metrics import patch_metrics, linear_categories_cmap, linear_categories_labels
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories_sidebyside

buffer_file = get_filename('g2_26729_gullies_and_roads_buffer_categories.tif')

ds1, _ = patch_metrics(buffer_file, outdir='/tmp', stub='test_w8', plot=False, save_csv=False, save_tif=False, max_shelterbelt_width=8)
ds2, _ = patch_metrics(buffer_file, outdir='/tmp', stub='test_l25', plot=False, save_csv=False, save_tif=False, min_shelterbelt_length=25)
visualise_categories_sidebyside(
    ds1['linear_categories'], ds2['linear_categories'],
    colormap=linear_categories_cmap, labels=linear_categories_labels,
    title1="max width=8", title2="min length=25"
)