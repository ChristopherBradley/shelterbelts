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


def worldcover_bbox(bbox=[147.735717, -42.912122, 147.785717, -42.862122], crs="EPSG:4326", outdir=".", stub="Test"):
    """Download worldcover data for the region of interest"""
    
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
    ds = odc.stac.load(items, crs=crs, bbox=bbox)
    ds_map = ds.isel(time=0)['map']

    filename = os.path.join(outdir, f"{stub}_worldcover.tif")
    ds_map.rio.to_raster(filename)
    print("Downloaded", filename)

    return ds_map
    


def worldcover_centerpoint(lat=-34.3890427, lon=148.469499, buffer=0.05, outdir=".", stub="Test"):
    """Download worldcover data for the region of interest"""
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    crs="EPSG:4326"
    ds_map = worldcover_bbox(bbox, crs, outdir, stub)
    return ds_map, bbox


def visualise_worldcover(ds, bbox, outdir=".", stub="Test"):
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


# %%time
if __name__ == '__main__':
    
    # Change directory to this repo
    import os, sys
    repo_name = "shelterbelts"
    if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
        repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
    elif os.path.basename(os.getcwd()) != repo_name:
        repo_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
    else:  # Already running locally from repo root
        repo_dir = os.getcwd()
    os.chdir(repo_dir)
    sys.path.append(repo_dir)
    print(f"Running from {repo_dir}")

    # Coords for Fulham: -42.887122, 147.760717
    lat=-42.887122
    lon=147.760717
    buffer=0.025
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    ds, bbox = worldcover_centerpoint(lat=lat, lon=lon, buffer=buffer, outdir="data", stub="Fulham")
    visualise_worldcover(ds, bbox)


