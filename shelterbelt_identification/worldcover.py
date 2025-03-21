# +
# Downloading WorldCover using the same input parameters as the global canopy height

# +
# Following this example to download from the microsoft planetary computer
# https://planetarycomputer.microsoft.com/dataset/esa-worldcover#Example-Notebook

# +
# # !pip install rich
# -

import planetary_computer
import pystac_client


bbox_of_interest = [33.984, 0.788, 34.902, 1.533]


# +
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["esa-worldcover"],
    bbox=bbox_of_interest,
)

items = list(search.get_items())
items

# +
import rich.table

# Assets
t_assets = rich.table.Table("Key", "Value")
for key, asset in items[0].assets.items():
    t_assets.add_row(key, asset.title)
t_assets
# -

# Metadata
t_metadata = rich.table.Table("Key", "Value")
for k, v in sorted(items[0].properties.items()):
    t_metadata.add_row(k, str(v))
t_metadata

# +
from IPython.display import Image

Image(url=items[0].assets["rendered_preview"].href)

# +
class_list = items[0].assets["map"].extra_fields["classification:classes"]
classmap = {
    c["value"]: {"description": c["description"], "hex": c["color-hint"]}
    for c in class_list
}

t = rich.table.Table("Value", "Description", "Hex Color")
for k, v in classmap.items():
    t.add_row(str(k), v["description"], v["hex"])
t

# +
import matplotlib.colors

colors = ["#000000" for r in range(256)]
for key, value in classmap.items():
    colors[int(key)] = f"#{value['hex']}"
cmap = matplotlib.colors.ListedColormap(colors)

# sequences needed for an informative colorbar
values = [key for key in classmap]
boundaries = [(values[i + 1] + values[i]) / 2 for i in range(len(values) - 1)]
boundaries = [0] + boundaries + [255]
ticks = [(boundaries[i + 1] + boundaries[i]) / 2 for i in range(len(boundaries) - 1)]
tick_labels = [value["description"] for value in classmap.values()]

# +
import odc.stac

ds = odc.stac.load(items, crs="EPSG:4326", resolution=0.0001, bbox=bbox_of_interest)
map_data = ds["map"].isel(time=-1).load()
map_data

# +
from matplotlib import cm
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 14))
normalizer = matplotlib.colors.Normalize(vmin=0, vmax=255)

map_data.isel(latitude=slice(3000, 6000), longitude=slice(4000, 7000)).plot(
    ax=ax, cmap=cmap, norm=normalizer
)

colorbar = fig.colorbar(
    cm.ScalarMappable(norm=normalizer, cmap=cmap),
    boundaries=boundaries,
    values=values,
    cax=fig.axes[1].axes,
)
colorbar.set_ticks(ticks, labels=tick_labels)

ax.set_axis_off()
ax.set_title("ESA WorldCover at Mount Elgin");
