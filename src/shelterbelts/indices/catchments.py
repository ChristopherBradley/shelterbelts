# +
# Code taken from my honours in 2021, updated during my phd in 2024 and 2025: 
# https://gitlab.com/civilise-ai/tl-2024/-/blob/main/dem-server/Backend/ridge_and_gully/catchments.py

# +
import os
import numpy as np
from scipy import ndimage
from rasterio.features import rasterize
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import LineString

from DAESIM_preprocess.topography import dirmap, pysheds_accumulation
from shelterbelts.apis.worldcover import tif_categorical
# -


import matplotlib.pyplot as plt

# +
# Monkey patch - their function seems to be broken in later versions of numpy because of line 1417 should be np.false_ instead of the python type False
import geojson
import pysheds._sgrid as _self
from pysheds.sview import View
from pysheds.grid import Grid

def patched_extract_river_network(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                            routing='d8', algorithm='iterative', **kwargs):
    
    if routing.lower() == 'd8':
        fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
    else:
        raise NotImplementedError('Only implemented for `d8` routing.')

    # Literally just changed this line from False to np.bool_(False)
    mask_overrides = {'dtype' : np.bool_, 'nodata' : np.bool_(False)}
    
    kwargs.update(fdir_overrides)
    fdir = self._input_handler(fdir, **kwargs)
    kwargs.update(mask_overrides)
    mask = self._input_handler(mask, **kwargs)
    nodata_cells = self._get_nodata_cells(fdir)
    invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
    fdir[nodata_cells] = 0
    fdir[invalid_cells] = 0
    maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=False)
    masked_fdir = np.where(mask, fdir, 0).astype(np.int64)
    startnodes = np.arange(fdir.size, dtype=np.int64)
    endnodes = _self._flatten_fdir_numba(masked_fdir, dirmap).reshape(fdir.shape)
    indegree = np.bincount(endnodes.ravel(), minlength=fdir.size).astype(np.uint8)
    orig_indegree = np.copy(indegree)
    startnodes = startnodes[(indegree == 0)]
    if algorithm.lower() == 'iterative':
        profiles = _self._d8_stream_network_iter_numba(endnodes, indegree,
                                                        orig_indegree, startnodes)
    elif algorithm.lower() == 'recursive':
        profiles = _self._d8_stream_network_recur_numba(endnodes, indegree,
                                                        orig_indegree, startnodes)
    else:
        raise ValueError('Algorithm must be `iterative` or `recursive`.')
    featurelist = []
    for index, profile in enumerate(profiles):
        yi, xi = np.unravel_index(list(profile), fdir.shape)
        x, y = View.affine_transform(self.affine, xi, yi)
        line = geojson.LineString(np.column_stack([x, y]).tolist())
        featurelist.append(geojson.Feature(geometry=line, id=index))
    geo = geojson.FeatureCollection(featurelist)
    return geo
    
Grid.extract_river_network = patched_extract_river_network


# -


def find_segment_above(acc, coord, branches_np):
    """Look for a segment upstream with the highest accumulation"""
    segment_above = None
    acc_above = -1
    for i, branch in enumerate(branches_np):
        if branch[-1] == coord:
            branch_acc = acc[branch[-2][0], branch[-2][1]] 
            if branch_acc > acc_above:
                segment_above = i
                acc_above = branch_acc
    return segment_above

def catchment_gullies(grid, fdir, acc, num_catchments=10):
    """Find the largest gullies"""

    # Extract the branches
    branches = grid.extract_river_network(fdir, acc > np.max(acc)/(num_catchments*10), dirmap=dirmap, nodata=np.float64(np.nan))

    # Convert the branches to numpy coordinates 
    branches_np = []
    for i, feature in enumerate(branches["features"]):
        line_coords = feature['geometry']['coordinates']
        branch_np = []
        for coord in line_coords:
            col, row = ~grid.affine * (coord[0], coord[1])
            row, col = int(round(row)), int(round(col))
            branch_np.append([row,col])
        branches_np.append(branch_np)

    # Repeatedly find the main segments to the branch with the highest accumulation. 
    full_branches = []
    for i in range(num_catchments):
        
        # Using the second last pixel before it's merged with another branch.
        branch_accs = [acc[branch[-2][0], branch[-2][1]] for branch in branches_np]
        largest_branch = np.argmax(branch_accs)

        # Follow the stream all the way up this branch
        branch_segment_ids = []
        while largest_branch != None:
            upper_coord = branches_np[largest_branch][0]
            branch_segment_ids.append(largest_branch)
            largest_branch = find_segment_above(acc, upper_coord, branches_np)

        # Combine the segments in this branch
        branch_segments = [branches_np[i] for i in sorted(branch_segment_ids)]
        branch_combined = [item for sublist in branch_segments for item in sublist]
        full_branches.append(branch_combined)

        # Remove all the segments from that branch and start again
        branch_segments_sorted = sorted(branch_segment_ids, reverse=True)
        for i in branch_segments_sorted:
            del branches_np[i]

    # Extract the gullies
    gullies = np.zeros(acc.shape, dtype=bool)
    for branch in full_branches:
        for x, y in branch:
            gullies[x, y] = True

    return gullies, full_branches


def catchment_ridges(grid, fdir, acc, full_branches):
    """Finds the ridges/catchment boundaries corresponding to those gullies"""

    # Progressively delineate each catchment
    catchment_id = 1
    all_catchments = np.zeros(acc.shape, dtype=int)
    for branch in full_branches:
        
        # Find the coordinate with second highest accumulation
        coords = branch[-2]

        # Convert from numpy coordinate to geographic coordinate
        x, y = grid.affine * (coords[1], coords[0])

        # Generate the catchment above that pixel
        catch = grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, 
                            xytype='coordinate', nodata_out=np.bool_(False))

        # Override relevant pixels in all_catchments with this new catchment_id
        all_catchments[catch] = catchment_id
        catchment_id += 1

    # Find the edges of the catchments
    sobel_x = ndimage.sobel(all_catchments, axis=0)
    sobel_y = ndimage.sobel(all_catchments, axis=1)  
    edges = np.hypot(sobel_x, sobel_y) 
    ridges = edges > 0

    return ridges


# +
outdir = '../../../outdir/'
stub = 'g2_26729'

filename_dem = os.path.join(outdir, f"{stub}_terrain.tif")
filename_hydrolines = os.path.join(outdir, f"{stub}_hydrolines_cropped.gpkg")
# -

# Load the dem
da = rxr.open_rasterio(filename_dem).isel(band=0).drop_vars('band')
ds = da.to_dataset(name='terrain')
ds.rio.write_crs("EPSG:3857", inplace=True)

# I might want to smooth the terrain tif like in topography.py if downloading from terrain_tiles before running the hydrology functions
from scipy.ndimage import gaussian_filter
terrain_tif = os.path.join(outdir, f"{stub}_terrain_smoothed.tif")
sigma = 5
dem = ds['terrain'].values
dem_smooth = gaussian_filter(dem.astype(float), sigma=sigma)
ds['dem_smooth'] = (["y", "x"], dem_smooth)
ds["dem_smooth"].rio.to_raster(terrain_tif)


# Generate the ridges and gullies 
# We actually already have the gullies from the hydrolines, so the ridges are all we care about
# Might want to adjust the full_branches input for catchment ridges to be based on the hydrolines instead of catchment_gullies?
grid, dem, fdir, acc = pysheds_accumulation(terrain_tif)


gullies, full_branches = catchment_gullies(grid, fdir, acc, num_catchments=10)


grid.crs

gdf_hydrolines = gpd.read_file(filename_hydrolines)
gdf_hydrolines_reprojected = gdf_hydrolines.to_crs(grid.crs)

# Rasterize the hydrolines
hydro_branches = []
for geom in gdf_hydrolines_reprojected.geometry:
    coords = geom.coords 
    branch = []
    for x, y in coords:
        col, row = ~grid.affine * (x, y)
        row, col = int(round(row)), int(round(col))
        branch.append([row, col])
    hydro_branches.append(branch)

hydro_branches

grid.affine

# Rasterize the hydrolines
shapes = [(geom, 1) for geom in gdf_hydrolines_reprojected.geometry]
hydro_gullies = rasterize(
    shapes,
    out_shape=acc.shape,
    transform=grid.affine, 
    fill=0
)

hydro_ridges = catchment_ridges(grid, fdir, acc, hydro_branches)

plt.imshow(hydro_ridges)

# Save the gullies and ridges as tifs for viewing in QGIS
ds['gullies'] = (["y", "x"], gullies)
ds['ridges'] = (["y", "x"], ridges)

ds['hydro_gullies'] = (["y", "x"], hydro_gullies)
ds['hydro_ridges'] = (["y", "x"], hydro_ridges)

gullies_cmap = {
    0: (255, 255, 255),
    1: (0, 0, 255),
}
ridges_cmap = {
    0: (255, 255, 255),
    1: (255, 0, 0),
}

filename_gullies = os.path.join(outdir, f"{stub}_gullies.tif")
filename_ridges = os.path.join(outdir, f"{stub}_ridges.tif")
filename_hydrogullies = os.path.join(outdir, f"{stub}_hydrogullies.tif")
filename_hydroridges = os.path.join(outdir, f"{stub}_hydroridges.tif")

tif_categorical(ds['gullies'], filename_gullies, colormap=gullies_cmap)

tif_categorical(ds['ridges'], filename_ridges, colormap=ridges_cmap)

tif_categorical(ds['hydro_gullies'], filename_hydrogullies, colormap=gullies_cmap)

tif_categorical(ds['hydro_ridges'], filename_hydroridges, colormap=ridges_cmap)
