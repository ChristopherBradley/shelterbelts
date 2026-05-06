from shelterbelts.indices.buffer_categories import buffer_categories, buffer_categories_cmap, buffer_categories_labels
from shelterbelts.utils.filepaths import get_filename
from shelterbelts.utils.visualisation import visualise_categories_sidebyside

cover_file = get_filename('g2_26729_cover_categories.tif')
gullies_file = get_filename('g2_26729_hydrolines.tif')
ridges_file = get_filename('g2_26729_DEM-S_ridges.tif')
roads_file = get_filename('g2_26729_roads.tif')

# buffer_width: 1 vs 5 pixels (gullies only)
ds1 = buffer_categories(cover_file, gullies_file, outdir='/tmp', stub='buf1', plot=False, savetif=False, buffer_width=1)
ds2 = buffer_categories(cover_file, gullies_file, outdir='/tmp', stub='buf5', plot=False, savetif=False, buffer_width=5)
visualise_categories_sidebyside(
    ds1['buffer_categories'], ds2['buffer_categories'],
    colormap=buffer_categories_cmap, labels=buffer_categories_labels,
    title1="buffer_width=1", title2="buffer_width=5"
)

# gullies+roads vs gullies+roads+ridges
ds_gul_roads = buffer_categories(cover_file, gullies_file, roads_data=roads_file, outdir='/tmp', stub='gul_roads', plot=False, savetif=False, buffer_width=3)
ds_all = buffer_categories(cover_file, gullies_file, ridges_data=ridges_file, roads_data=roads_file, outdir='/tmp', stub='gul_rid_roads', plot=False, savetif=False, buffer_width=3)
visualise_categories_sidebyside(
    ds_gul_roads['buffer_categories'], ds_all['buffer_categories'],
    colormap=buffer_categories_cmap, labels=buffer_categories_labels,
    title1="Gullies + Roads", title2="Gullies + Roads + Ridges"
)