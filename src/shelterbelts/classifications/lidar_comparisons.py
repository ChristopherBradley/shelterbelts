import geopandas as gpd

filename = '/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/tiff_centroids_years.gpkg'

# !ls /Users/christopherbradley/Documents/PHD/Data/Nick_outlines/tiff_centroids_years.gpkg

gdf_centroids = gpd.read_file(filename) # This gives some error messages, but actually works - they should just be warnings.

interesting_tile_ids = [
    "g2_017_",
    "g2_019_",
    "g2_021_",
    "g2_21361_",
    "g2_23939_",
    "g2_23938_",
    "g2_09_",
    "g2_2835_",
    "g2_25560_",
    "g2_24903_"
]

filenames = [f'{tile_id}binary_tree_cover_10m.tiff' for tile_id in interesting_tile_ids]

gdf_interesting = gdf_centroids[gdf_centroids['filename'].isin(filenames)]

filename = '~/Desktop/centroids_interesting.gpkg'
gdf_interesting.to_file(filename)






