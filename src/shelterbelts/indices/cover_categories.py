
import os
import argparse

import xarray as xr
import rioxarray as rxr

from shelterbelts.apis.worldcover import worldcover_cmap, worldcover_labels, tif_categorical, visualise_categories
from shelterbelts.indices.tree_categories import tree_categories_cmap, tree_categories_labels

# +

cover_cmap = {
    31: (203, 219, 115),
    32: (255, 255, 76),
    41: (146, 104, 143),
    42: (240, 150, 255)
}

cover_labels = {
    31: "Unsheltered Grassland",
    32: "Sheltered Grassland",
    41: "Unsheltered Cropland",
    42: "Sheltered Cropland"
}

cover_categories_labels = tree_categories_labels | worldcover_labels | cover_labels
cover_categories_cmap = tree_categories_cmap | worldcover_cmap | cover_cmap
inverted_labels = {v: k for k, v in cover_categories_labels.items()}


def cover_categories(shelter_tif, worldcover_tif, outdir='.', stub='TEST', savetif=True, plot=True):
    """Reclassify non-tree pixels with categories from worldcover
    
    Parameters
    ----------
        category_tif: Integer tif file generated by shelter_categories.py
        worldcover_tif: Integer tif file generated by apis.worldcover.py
            
    Returns
    -------
        ds: an xarray with a band 'cover_categories', where the integers represent the categories defined in 'cover_category_labels'.

    Downloads
    ---------
        cover_categories.tif: A tif file of the 'cover_categories' band in ds, with colours embedded.
        cover_categories.png: A png file like the tif file, but with a legend as well.
    
    """
    da_shelter = rxr.open_rasterio(shelter_tif).squeeze('band').drop_vars('band')
    da_worldcover = rxr.open_rasterio(worldcover_tif).squeeze('band').drop_vars('band')

    da_worldcover2 = da_worldcover.rio.reproject_match(da_shelter)

    # Unfortunately this removes the crs
    da_override_trees = xr.where((da_shelter >= 10) & (da_shelter < 20), da_shelter, da_worldcover2)
    # I could do it like this, but I think this is less readable
    # da_override_trees = da_shelter.where((da_shelter >= 10) & (da_shelter < 20), da_shelter, da_worldcover2)

    # Reassign pixels labelled by worldcover as tree but the shelter_tif as not tree, into grassland
    # I'm assuming that the shelter_tif is more accurate than the worldcover_tif in terms of classifying tree vs no tree
    da_override_grass = xr.where((da_override_trees == 10), 30, da_override_trees)

    # I should probably be using the inverted_labels with strings instead of numbers for all of these
    sheltered_grass = (da_override_grass == 30) & (da_shelter == 2)
    unsheltered_grass = (da_override_grass == 30) & (da_shelter == 0)
    sheltered_crop = (da_override_grass == 40) & (da_shelter == 2)
    unsheltered_crop = (da_override_grass == 40) & (da_shelter == 0)

    da = da_override_grass
    da = xr.where(sheltered_grass, inverted_labels["Sheltered Grassland"], da)
    da = xr.where(unsheltered_grass, inverted_labels["Unsheltered Grassland"], da)
    da = xr.where(sheltered_crop, inverted_labels["Sheltered Cropland"], da)
    da = xr.where(unsheltered_crop, inverted_labels["Unsheltered Cropland"], da)

    ds = da.to_dataset(name='cover_categories')

    # Reassign the crs 
    ds.rio.write_crs(da_shelter.rio.crs, inplace=True)

    if savetif:
        filename = os.path.join(outdir,f"{stub}_cover_categories.tif")
        tif_categorical(ds['cover_categories'], filename, cover_categories_cmap)

    if plot:
        filename_png = os.path.join(outdir, f"{stub}_cover_categories.png")
        visualise_categories(ds['cover_categories'], filename_png, cover_categories_cmap, cover_categories_labels, "Cover Categories")

    ds = ds.rename({'x':'longitude', 'y': 'latitude'})
    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--shelter_tif', help='Integer tif file generated by shelter_categories.py')
    parser.add_argument('--worldcover_tif', help='Integer tif file generated by apis.worldcover.py')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', default='TEST', help='Prefix for output files.')
    parser.add_argument('--plot', default=False, action="store_true", help="Boolean to Save a png file along with the tif")

    return parser.parse_args()

# -

if __name__ == '__main__':
    args = parse_arguments()

    shelter_tif = args.shelter_tif
    worldcover_tif = args.worldcover_tif
    outdir = args.outdir
    stub = args.stub
    plot = args.plot

    ds = cover_categories(shelter_tif, worldcover_tif, outdir, stub, savetif=True, plot=plot)
