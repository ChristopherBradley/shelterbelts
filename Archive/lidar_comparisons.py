import geopandas as gpd
import rioxarray as rxr
from sklearn.metrics import classification_report


# Used this code to create centroids of tiles I identified earlier as having a good distribution of tree and non-tree pixels
def interesting_centroids():
    filename = '/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/tiff_centroids_years.gpkg'
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


tif_nick = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_26729/g2_26729_binary_tree_cover_10m_Nick.tiff'
tif_feb = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_26729/Feb_g2_26729_woodyveg_res10_height2m.tif'
tif_sep = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_26729/Sep_g2_26729_woodyveg_res10_height2m.tif'
tif_nsw = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_26729/Sep_NSW_g2_26729_woodyveg_res10_cat5.tif'

da_nick = rxr.open_rasterio(tif_nick).isel(band=0).drop_vars('band')
da_feb = rxr.open_rasterio(tif_feb).isel(band=0).drop_vars('band')
da_sep = rxr.open_rasterio(tif_sep).isel(band=0).drop_vars('band')
da_nsw = rxr.open_rasterio(tif_nsw).isel(band=0).drop_vars('band')


def tif_comparison(da1, da2):
    """Create a classification report comparing the two rasters"""

    # reproject the larger tif to match the smaller tif
    if da1.shape[0] >= da2.shape[0] and da1.shape[1] >= da2.shape[1]:
        da1 = da1.rio.reproject_match(da2)
    elif da2.shape[0] >= da1.shape[0] and da2.shape[1] >= da1.shape[1]:
        da2 = da2.rio.reproject_match(da1)
    else:
        print("Unimplemented: Shapes are overlaying in an annoying way. Should take the intersection.")

    print(classification_report(da1.values.flatten(), da2.values.flatten()))
    


tif_comparison(da_nick, da_feb)

tif_comparison(da_nick, da_sep)

tif_comparison(da_nsw, da_sep)

tif_comparison(da_feb, da_sep)

# +
nick_act = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_09/g2_09_binary_tree_cover_10m_Nick.tiff'
act_2015_mine = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_09/2015_g2_09_woodyveg_res10_height2m.tif'
act_2015_theirs = '/Users/christopherbradley/Documents/PHD/Data/ELVIS/tif_comparisons/g2_09/ACT2015_g2_09_woodyveg_res10_cat5.tif'


# -

da_nick_act = rxr.open_rasterio(nick_act).isel(band=0).drop_vars('band')
da_act_2015_mine= rxr.open_rasterio(act_2015_mine).isel(band=0).drop_vars('band')
da_act_2015_theirs = rxr.open_rasterio(act_2015_theirs).isel(band=0).drop_vars('band')


da_act_2015_mine = da_act_2015_mine.rio.write_crs('EPSG:28355')
da_act_2015_theirs = da_act_2015_theirs.rio.write_crs('EPSG:28355')

tif_comparison(da_act_2015_mine, da_act_2015_theirs)

tif_comparison(da_nick_act, da_act_2015_mine)

tif_comparison(da_nick_act, da_act_2015_theirs)
