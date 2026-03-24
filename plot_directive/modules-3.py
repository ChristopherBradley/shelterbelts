from shelterbelts.indices.cover_categories import cover_categories, cover_categories_cmap, cover_categories_labels
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories

shelter_file = get_filename('g2_26729_shelter_categories.tif')
worldcover_file = get_filename('g2_26729_worldcover.tif')

ds_cover = cover_categories(shelter_file, worldcover_file, outdir='/tmp', plot=False, savetif=False)
visualise_categories(
    ds_cover['cover_categories'],
    colormap=cover_categories_cmap, labels=cover_categories_labels,
    title="Cover Categories"
)