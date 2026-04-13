import rioxarray as rxr
from shelterbelts.indices.opportunities import opportunities, opportunity_cmap, opportunity_labels
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories_sidebyside

tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
roads_file = get_filename('g2_26729_roads.tif')
gullies_file = get_filename('g2_26729_hydrolines.tif')
dem_file = get_filename('g2_26729_DEM-H.tif')
worldcover_file = get_filename('g2_26729_worldcover.tif')
common = dict(dem_data=dem_file, worldcover_data=worldcover_file, outdir='/tmp', plot=False, savetif=False)

da_zero = rxr.open_rasterio(roads_file).isel(band=0).drop_vars('band') * 0  # Maybe gullies_data = None should mean no gullies instead of autogenerating the gullies?

ds_roads = opportunities(tree_file, roads_data=roads_file, gullies_data=da_zero, **common, contour_spacing=0)
ds_gullies = opportunities(tree_file, roads_data=da_zero, gullies_data=gullies_file, **common, contour_spacing=0)
visualise_categories_sidebyside(
    ds_roads['opportunities'], ds_gullies['opportunities'],
    colormap=opportunity_cmap, labels=opportunity_labels,
    title1="Just roads", title2="Just gullies"
)

ds_w1 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, width=1)
ds_w5 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, width=5)
visualise_categories_sidebyside(
    ds_w1['opportunities'], ds_w5['opportunities'],
    colormap=opportunity_cmap, labels=opportunity_labels,
    title1="width=1", title2="width=5"
)

ds_cs5 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, contour_spacing=5)
ds_cs20 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, contour_spacing=20)
visualise_categories_sidebyside(
    ds_cs5['opportunities'], ds_cs20['opportunities'],
    colormap=opportunity_cmap, labels=opportunity_labels,
    title1="contour_spacing=5", title2="contour_spacing=20"
)