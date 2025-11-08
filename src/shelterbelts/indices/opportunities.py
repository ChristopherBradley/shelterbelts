# +
import numpy as np
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box      
from scipy.ndimage import binary_dilation

import networkx as nx
from collections import Counter
from skimage.measure import label
from skimage.morphology import skeletonize

import matplotlib.pyplot as plt

# from shelterbelts.apis.worldcover import worldcover_bbox, tif_categorical
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds

# -

from shelterbelts.indices.full_pipelines import worldcover_dir, hydrolines_gdb, roads_gdb


def segmentation(da, min_branch_length=10):
    """Converts a binary skeletonized da into an integer segmented da"""
    # River segmentation algorithm (entirely ChatGPT)

    # Get indices of river pixels
    river_pixels = np.argwhere(river_mask == 1)
    
    # Create graph where each river pixel is a node
    G = nx.Graph()
    
    for y, x in river_pixels:
        G.add_node((y, x))
        # Check 8-connectivity neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < river_mask.shape[0] and 0 <= nx_ < river_mask.shape[1]:
                    if river_mask[ny, nx_] == 1:
                        G.add_edge((y, x), (ny, nx_))
    
    # Identify junctions (>=3 neighbors) and endpoints (==1 neighbor)
    junctions = {n for n, d in G.degree() if d >= 3}
    endpoints = {n for n, d in G.degree() if d == 1}
    
    # Extract branches: paths between junctions and endpoints
    branches = []
    visited_edges = set()
    
    for start in endpoints | junctions:
        for neighbor in G.neighbors(start):
            edge = frozenset([start, neighbor])
            if edge in visited_edges:
                continue
    
            branch = [start, neighbor]
            visited_edges.add(edge)
            prev, current = start, neighbor
    
            # Walk until we hit a junction or endpoint
            while current not in (endpoints | junctions):
                next_nodes = [n for n in G.neighbors(current) if n != prev]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = frozenset([current, next_node])
                if edge in visited_edges:
                    break
                branch.append(next_node)
                visited_edges.add(edge)
                prev, current = current, next_node
    
            branches.append(branch)
    
    # num_branches = len(branches)
    # print(f"Number of river branches: {num_branches}")

    # Create a raster
    branch_labels = np.zeros_like(river_mask, dtype=np.int32)
    for i, branch in enumerate(branches, 1):
        for (y, x) in branch:
            branch_labels[y, x] = i

    # Find the small branches that aren't really branches
    branch_sizes = Counter(branch_labels.ravel())
    del branch_sizes[0] 
    branch_labels_new = branch_labels.copy()
    small_branches = [bid for bid, size in branch_sizes.items() if size < min_branch_length]
    
    # Merge small branches into the larger ones
    while True:
        branch_sizes = Counter(branch_labels_new.ravel())
        branch_sizes.pop(0, None)  # remove background
    
        small_branches = [bid for bid, size in branch_sizes.items() if size < min_branch_length]
        if not small_branches:
            break  # already merged all the branches
    
        merged_this_round = False
    
        for bid in small_branches:
            mask = branch_labels_new == bid
            if not np.any(mask):
                continue  # already merged this branch
    
            # Find neighboring branches
            dilated = binary_dilation(mask, structure=np.ones((3, 3)))
            neighbors = np.unique(branch_labels_new[dilated & (branch_labels_new != bid) & (branch_labels_new != 0)])
    
            if len(neighbors) == 0:
                continue
    
            # Merge into largest neighboring branch
            largest_neighbor = max(neighbors, key=lambda n: branch_sizes.get(n, 0))
            branch_labels_new[mask] = largest_neighbor
            merged_this_round = True
    
        if not merged_this_round:
            break  # already merged all the branches

    return branch_labels_new



def opportunities(da_trees, da_roads, da_gullies, da_ridges, da_worldcover, outdir='.', stub='TEST', tmpdir='.', width=3, contour_spacing=30):
    """
    Parameters
    ----------
        da_trees, da_roads, da_gullies, da_ridges: binary xarrays 
        da_worldcover: Int xarray for grass and crop categories
        outdir: The output directory to save the results.
        stub: Prefix for output files. If not specified, then it appends 'categorised' to the original filename.
        road_width: Number of pixels away from the feature that still counts as within the buffer
            - May want different widths for different buffers later
        contour_spacing: Number of pixels between each contour
        
    Returns
    -------
        ds: an xarray with a band 'opportunities', where the integers represent the categories defined in 'opportunity_labels'.
    
    Downloads
    ---------
        opportunities.tif: A tif file of the 'opportunities' band in ds, with colours embedded.
    """


# +
stub='TEST'
tmpdir = '/scratch/xe2/cb8590/'
buffer_width=3

percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2024_actnsw_4326/subfolders/lat_34_lon_140/34_13-141_90_y2024_predicted.tif' # Should be fine
da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
da_trees = da_percent > 50
da_trees = da_trees.astype('uint8')
ds_woody_veg = da_trees.to_dataset(name='woody_veg')

gdf_hydrolines, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent)
da_hydrolines = ds_hydrolines['gullies']
gdf_roads, ds_roads = hydrolines(None, roads_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent, layer='NationalRoads_2025_09')
da_roads = ds_roads['gullies']

gs_bounds = gpd.GeoSeries([box(*da_trees.rio.bounds())], crs=da_trees.rio.crs)
bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
worldcover_geojson = 'cb8590_Worldcover_Australia_footprints.gpkg'
worldcover_stub = f'TEST' # Anything that might be run in parallel needs a unique filename, so we don't get rasterio merge conflicts
mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, worldcover_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
da_worldcover2 = da_worldcover.rio.reproject_match(da_trees) # Should do this within full_pipelines so it doesn't need to happen twice


# +
# Find options for buffered gullies and roads that currently cropland or grassland
grass_crops = (da_worldcover2 == 30) | (da_worldcover2 == 40)

y, x = np.ogrid[-buffer_width:buffer_width+1, -buffer_width:buffer_width+1]
gap_kernel = (x**2 + y**2 <= buffer_width**2)
buffered_gullies = binary_dilation(da_hydrolines.values, structure=gap_kernel)
gully_opportunities = buffered_gullies & grass_crops & ~da_trees

buffered_roads = binary_dilation(da_roads.values, structure=gap_kernel)
road_opportunities = buffered_roads & grass_crops & ~da_trees & ~gully_opportunities  # Prioritising gullies over roads

# Create a layer that combines these opportunities
plt.imshow(road_opportunities)
# -
# (Quickly) load the dem for this area. 
dem_dir = '/g/data/xe2/cb8590/NSW_5m_DEMs_3857'
dem_gpkg = 'cb8590_NSW_5m_DEMs_3857_footprints.gpkg'


# +
# from shelterbelts.classifications.bounding_boxes import bounding_boxes
# gdf = bounding_boxes(dem_dir, crs='EPSG:3857') # Took 30 secs
# -

# %%time
dem_stub = 'TEST'
bbox_3857 = list(gs_bounds.to_crs('EPSG:3857').bounds.iloc[0])
mosaic, out_meta = merge_tiles_bbox(bbox_3857, tmpdir, dem_stub, dem_dir, dem_gpkg, 'filename', verbose=True) 
ds_dem = merged_ds(mosaic, out_meta, 'dem')
ds_dem['dem'].plot()

da_hydrolines.plot()

river_mask = da_hydrolines.values

river_mask = skeletonize(da_hydrolines.values)


# %%time
branch_labels_new = segmentation(river_mask)


plt.imshow(branch_labels_new)

ds_woody_veg['branch_labels_new'] = ('y', 'x'), branch_labels_new
ds_woody_veg['branch_labels_new'].astype(float).rio.to_raster('/scratch/xe2/cb8590/tmp/branch_labels_new.tif')

