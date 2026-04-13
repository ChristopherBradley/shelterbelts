from shelterbelts.apis.worldcover import worldcover, worldcover_cmap, worldcover_labels
from shelterbelts.utils.visualisation import visualise_categories

ds = worldcover(buffer=0.01, save_tif=False, plot=False)
visualise_categories(ds['worldcover'], colormap=worldcover_cmap, labels=worldcover_labels, title="ESA WorldCover")