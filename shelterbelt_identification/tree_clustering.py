# +
# Group trees together and calculate statistics including: 
# Number of shelterbelts w/ length, width, area, perimeter, height (min, mean, max) for each (and then mean and sd)
# Area of sheltered and unsheltered crop & pasture by region and different thresholds
# -
import os
import glob
import pickle
import numpy as np
import xarray as xr
import rioxarray as rxr
import pyproj


# Load the woody veg or canopy cover tiff file
outdir = "../data/"
filename = os.path.join(outdir, "Tas_WoodyVeg_201903_v2.2.tif")  # Binary classifications
sub_stub = "woodyveg"
ds_original = rxr.open_rasterio(filename)

# Create a 5km x 5km bounding box
lon, lat = 147.4793, -42.3906
buffer_m = 2500
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True).transform
x, y = project(lon, lat)
bbox = x - buffer_m, y + buffer_m, x + buffer_m, y - buffer_m  # Not sure why the miny requires addition instead of subtraction, but this gives the 5km x 5km region
minx, miny, maxx, maxy = bbox
bbox

# Select the 5km x 5km region
da = ds_original.sel(band=1, x=slice(minx, maxx), y=slice(miny, maxy))

# Code from civilise.ai that I wrote during honours. There's probably a better way to do this now. 
adjacencies = np.array([(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)])
direct_adjacencies = np.array([(0, -1),  (1, 0),  (0, 1),  (-1, 0)])
def group_cells(bool_array):
    """Group the cells. Each group is a dictionary with the cell coord being the key and the value being a list of cells (initially empty)"""
    xs, ys = np.where(bool_array)
    coords = [(x,y) for x, y in zip(xs,ys)]
    groups = dict()
    for coord in coords:
        # Find all the neighbours of the coord
        neighbours = adjacencies + coord
        neighbours = [tuple(n) for n in neighbours]
        assigned_groups = set()
        for neighbour in neighbours:
            for group in groups.keys():
                if neighbour in groups[group].keys():
                    # A new cell may be attached to two different groups
                    assigned_groups.add(group)
        assigned_groups = list(assigned_groups)
        # Create a new group
        if len(assigned_groups) == 0:
            if len(groups) == 0:
                new_group = 0
            else:
                new_group = max(groups.keys()) + 1
            groups[new_group] = {coord:[]}
        # Simply add this cell to the other group
        if len(assigned_groups) == 1:
            groups[assigned_groups[0]][coord] = []
        # Combine the groups together
        if len(assigned_groups) > 1:
            combined_group = {coord:[]}
            assigned_groups = sorted(assigned_groups, reverse=True)
            for group in assigned_groups:
                combined_group = {**combined_group, **groups[group]}
                del groups[group]
            new_group = 0 if len(groups) == 0 else max(groups.keys()) + 1
            groups[new_group] = combined_group
    return groups



woody_veg = np.array(da.values - 1, dtype = bool)

# %%time
groups = group_cells(woody_veg)
# 27 secs for a 5km x 5km area

# Just need list of coords per group for my purposes right now.
group_coords = {list(group.keys()) for group in groups.values()]

# +
# Should add an area, length, width to each shelterbelt 
# Should filter out any shelterbelts smaller than a certain size
# -

# Add a group id to each pixel in woodyveg
shelterbelts = np.zeros(woody_veg.shape, dtype=float)  # For some reason the export to tif doesn't work if I specify int
for group_idx, coords in enumerate(group_coords):
    for x, y in coords:
        shelterbelts[x, y] = group_idx

# Add these groups to the original xarray
da_reset = da.reset_coords("band", drop=True)
ds = da_reset.to_dataset(name="woody_veg")
da_shelterbelts = xr.DataArray(
    shelterbelts,
    dims=["y", "x"],
    coords={"x": ds.x, "y": ds.y},
    name="shelterbelts"
)
ds["shelterbelts"] = da_shelterbelts

ds['shelterbelts'].rio.to_raster("../data/test_shelterbelts.tif")
