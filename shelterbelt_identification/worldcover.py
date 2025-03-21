# +
# Example code is here: https://planetarycomputer.microsoft.com/dataset/esa-worldcover#Example-Notebook

# +
import os
import odc.stac
import pystac_client
import planetary_computer
import rioxarray as rxr

import matplotlib.colors
from matplotlib import cm
import matplotlib.pyplot as plt
# -

world_cover_layers = {
    "Tree cover": 10, # Green
    "Shrubland": 20, # Orange
    "Grassland": 30, # Yellow
    "Cropland": 40, # pink
    "Built-up": 50, # red
    "Permanent water bodies": 80, # blue
}


def worldcover(lat=-34.3890427, lon=148.469499, buffer=0.05, outdir=".", stub="Test"):
    """Download worldcover data for the region of interest"""
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox,
    )
    items = list(search.items())
    items = [items[0]]
    ds = odc.stac.load(items, crs="EPSG:4326", bbox=bbox)
    ds_map = ds.isel(time=0)['map']

    filename = os.path.join(outdir, f"{stub}_worldcover.tif")
    ds_map.rio.to_raster(filename)
    print("Downloaded", filename)
    
    return ds_map


def visualise_worldcover(ds, outdir=".", stub="Test"):
    """Pretty visualisation using the worldcover colour scheme"""
    
    # Download the colour scheme
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox,
    )
    items = list(search.items())
    items = [items[0]]
    
    class_list = items[0].assets["map"].extra_fields["classification:classes"]
    classmap = {
        c["value"]: {"description": c["description"], "hex": c["color-hint"]}
        for c in class_list
    }

    # Prep the colour bar
    colors = ["#000000" for r in range(256)]
    for key, value in classmap.items():
        colors[int(key)] = f"#{value['hex']}"
    cmap = matplotlib.colors.ListedColormap(colors)
    
    values = [key for key in classmap]
    boundaries = [(values[i + 1] + values[i]) / 2 for i in range(len(values) - 1)]
    boundaries = [0] + boundaries + [255]
    ticks = [(boundaries[i + 1] + boundaries[i]) / 2 for i in range(len(boundaries) - 1)]
    tick_labels = [value["description"] for value in classmap.values()]
    
    normalizer = matplotlib.colors.Normalize(vmin=0, vmax=255)

    # Plot the Map
    fig, ax = plt.subplots(figsize=(16, 14))
    ds.plot(
        ax=ax, cmap=cmap, norm=normalizer
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=normalizer, cmap=cmap),
        boundaries=boundaries,
        values=values,
        cax=fig.axes[1].axes,
    )
    colorbar.set_ticks(ticks, labels=tick_labels)

    filename = os.path.join(outdir, stub)
    plt.savefig(filename)
    print("Saved", filename)


if __name__ == '__main__':
    ds = worldcover()
    visualise_worldcover(ds)
