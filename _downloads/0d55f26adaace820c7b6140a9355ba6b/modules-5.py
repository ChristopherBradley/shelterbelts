from shelterbelts.indices.shelter_categories import shelter_categories, shelter_categories_cmap, shelter_categories_labels
from shelterbelts.utils.filepaths import get_filename, get_example_tree_categories_data
from shelterbelts.utils.visualisation import visualise_categories_sidebyside

ds_cat = get_example_tree_categories_data()
wind_file = get_filename('g2_26729_barra_daily.nc')

# density_threshold: 3 vs 10 (density method, no wind data)
ds1 = shelter_categories(ds_cat, outdir='/tmp', stub='dens1', plot=False, savetif=False, density_threshold=3)
ds2 = shelter_categories(ds_cat, outdir='/tmp', stub='dens2', plot=False, savetif=False, density_threshold=10)
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="density_threshold=3", title2="density_threshold=10"
)

# wind_method: MOST_COMMON vs WINDWARD
ds1 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wind1', plot=False, savetif=False, wind_method='MOST_COMMON')
ds2 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wind2', plot=False, savetif=False, wind_method='WINDWARD')
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="wind_method=MOST_COMMON", title2="wind_method=WINDWARD"
)

# distance_threshold: 10 vs 30 (with wind data)
ds1 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='dist1', plot=False, savetif=False, distance_threshold=10, wind_method='MOST_COMMON')
ds2 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='dist2', plot=False, savetif=False, distance_threshold=30, wind_method='MOST_COMMON')
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="distance_threshold=10", title2="distance_threshold=30"
)

# wind_threshold: 10 vs 30 (km/h)
ds1 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wt1', plot=False, savetif=False, wind_threshold=10, wind_method='MOST_COMMON')
ds2 = shelter_categories(ds_cat, wind_data=wind_file, outdir='/tmp', stub='wt2', plot=False, savetif=False, wind_threshold=30, wind_method='MOST_COMMON')
visualise_categories_sidebyside(
    ds1['shelter_categories'], ds2['shelter_categories'],
    colormap=shelter_categories_cmap, labels=shelter_categories_labels,
    title1="wind_threshold=10", title2="wind_threshold=30"
)