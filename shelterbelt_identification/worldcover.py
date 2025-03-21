# +
# Example code is here: https://planetarycomputer.microsoft.com/dataset/esa-worldcover#Example-Notebook
# -

import os
import odc.stac
import pystac_client
import planetary_computer

world_cover_layers = {
    "Tree cover": 10, # Green
    "Shrubland": 20, # Orange
    "Grassland": 30, # Yellow
    "Cropland": 40, # pink
    "Built-up": 50, # red
    "Permanent water bodies": 80, # blue
}


def worldcover(lat=-34.3890427, lon=148.469499, buffer=0.005, outdir=".", stub="Test"):
    """Download worldcover data for the region of interest"""
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox_of_interest,
    )
    items = list(search.items())
    items = [items[0]]
    ds = odc.stac.load(items, crs="EPSG:4326", bbox=bbox_of_interest)
    ds_map = ds.isel(time=0)['map']

    filename = os.path.join(outdir, f"{stub}_worldcover.tif")
    ds_map.rio.to_raster(filename)
    print("Downloaded", filename)
    
    return ds_map


if __name__ == '__main__':
    worldcover()


