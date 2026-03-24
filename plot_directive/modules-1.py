from shelterbelts.indices.tree_categories import tree_categories, tree_categories_cmap, tree_categories_labels
from shelterbelts.utils.visualisation import visualise_categories_sidebyside
from shelterbelts.utils.filepaths import get_filename

test_filename = get_filename('g2_26729_binary_tree_cover_10m.tiff')

# edge_size: 1 vs 5
ds1 = tree_categories(test_filename, edge_size=1)
ds2 = tree_categories(test_filename, edge_size=5)
visualise_categories_sidebyside(
    ds1['tree_categories'], ds2['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="edge_size=1", title2="edge_size=5"
)

# min_patch_size: 10 vs 30
ds1 = tree_categories(test_filename, min_patch_size=10)
ds2 = tree_categories(test_filename, min_patch_size=30)
visualise_categories_sidebyside(
    ds1['tree_categories'], ds2['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="min_patch_size=10", title2="min_patch_size=30"
)

# max_gap_size: 0 vs 2
ds1 = tree_categories(test_filename, max_gap_size=0)
ds2 = tree_categories(test_filename, max_gap_size=2)
visualise_categories_sidebyside(
    ds1['tree_categories'], ds2['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="max_gap_size=0", title2="max_gap_size=2"
)

# strict_core_area: False vs True
ds1 = tree_categories(test_filename, strict_core_area=False)
ds2 = tree_categories(test_filename, strict_core_area=True)
visualise_categories_sidebyside(
    ds1['tree_categories'], ds2['tree_categories'],
    colormap=tree_categories_cmap, labels=tree_categories_labels,
    title1="strict_core_area=False", title2="strict_core_area=True"
)