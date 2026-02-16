from shelterbelts.apis.worldcover import worldcover, worldcover_cmap, worldcover_labels
from shelterbelts.utils.visualisation import visualise_categories

ds_10km = worldcover(buffer=0.05)
print("worldcover coordinates:", list(ds_10km['worldcover'].coords.keys()))
visualise_categories(ds_10km['worldcover'], colormap=worldcover_cmap, labels=worldcover_labels, title="Larger Buffer")

from shelterbelts.indices.cover_categories import cover_categories, cover_categories_cmap, cover_categories_labels
from shelterbelts.utils.filepaths import get_filename

shelter_file = get_filename('g2_26729_shelter_categories.tif')
worldcover_file = get_filename('g2_26729_worldcover.tif')

ds_cover = cover_categories(shelter_file, worldcover_file, outdir='/tmp', plot=False, savetif=False)
print("cover_categories coordinates:", list(ds_cover['cover_categories'].coords.keys()))
visualise_categories(
    ds_cover['cover_categories'],
    colormap=cover_categories_cmap, labels=cover_categories_labels,
    title="Cover Categories"
)
print("Done")