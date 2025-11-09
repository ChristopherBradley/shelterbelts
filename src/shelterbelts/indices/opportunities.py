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
from skimage.measure import find_contours

import matplotlib.pyplot as plt

# from shelterbelts.apis.worldcover import worldcover_bbox, tif_categorical
from shelterbelts.apis.hydrolines import hydrolines
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds
from shelterbelts.apis.catchments import catchments  # This takes a while to import

# -

from shelterbelts.indices.full_pipelines import worldcover_dir, worldcover_geojson, hydrolines_gdb, roads_gdb

# +
nsw_dem_dir = '/g/data/xe2/cb8590/NSW_5m_DEMs_3857'
nsw_dem_gpkg = 'cb8590_NSW_5m_DEMs_3857_footprints.gpkg'

# Used this to prepare the nsw_dem_dir for quick merging (took 30 secs)
# from shelterbelts.classifications.bounding_boxes import bounding_boxes

# +
def segmentation(river_pixels, min_branch_length=10):
    """Converts a binary skeletonized array into an integer segmented array"""
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

    # Make the branch id's consecutive
    unique_vals, branch_labels_consecutive = np.unique(branch_labels_new, return_inverse=True)
    branch_labels_consecutive = branch_labels_consecutive.reshape(branch_labels_new.shape)
    
    return branch_labels_consecutive

def dict_to_grid(Z, dict_Contours, contours):
    # Function by Yasar in 2021
    Binary_Grid = np.zeros_like(Z)
    for height in contours:
        height_Contours = dict_Contours[f"{height:.2f}"]  # This means we can have more precision than just integer contours, for flattish areas
        for height_Contour in height_Contours:
            id_x = height_Contour[:,0].astype(np.uint64)
            id_y = height_Contour[:,1].astype(np.uint64)
            Binary_Grid[id_x, id_y] = True
    return Binary_Grid

def contours_interval(Z, interval=10, min_contour_length=100):
    """Create a binary raster along specific contours for consistency when running across multiple tiles"""
    Z = np.array(np.round(Z))
    minZ, maxZ = int(np.min(Z)), int(np.max(Z))

    contours = [height for height in range(minZ, maxZ) if height % interval == 0]
    
    dict_Contours = dict()
    for height in contours:
        contours_at_height = find_contours(Z, height)

        # Filter out small contours
        contours_at_height = [c for c in contours_at_height if len(c) >= min_contour_length]

        dict_Contours[f"{height:.2f}"] = contours_at_height
    
    grid = dict_to_grid(Z, dict_Contours, contours)
    
    return grid

def contours_equal_area(Z, num_contours=4, min_contour_length=100):
    """Creates a binary raster with contour intervals that split the landscape into equal-area bands."""
    Z = np.array(Z, dtype=float)
    Z_flat = Z.flatten()

    # Compute elevation thresholds that divide the data into equal-area (equal pixel count) bands
    percentiles = np.linspace(0, 100, num_contours + 2)[1:-1]  # exclude 0 and 100
    contour_heights = np.percentile(Z_flat, percentiles)

    dict_Contours = {}
    for h in contour_heights:
        contours_at_height = find_contours(Z, h)

        # Filter out small contours
        contours_at_height = [c for c in contours_at_height if len(c) >= min_contour_length]

        dict_Contours[f"{h:.2f}"] = contours_at_height

    grid = dict_to_grid(Z, dict_Contours, contour_heights)
    return grid


# +
def opportunities_da(da_trees, da_roads, da_gullies, da_ridges, da_contours, da_worldcover, outdir='.', stub='TEST', tmpdir='.', 
                     width=3):
    """
    Parameters
    ----------
        da_trees, da_roads, da_gullies, da_ridges, da_contours: binary xarrays 
        da_worldcover: Int xarray for grass and crop categories
        outdir: The output directory to save the results.
        stub: Prefix for output files. If not specified, then it appends 'categorised' to the original filename.
        width: Number of pixels away from the feature that still counts as within the buffer
            - May want different widths for different buffers later
        
    Returns
    -------
        ds: an xarray with a band 'opportunities', where the integers represent the categories defined in 'opportunity_labels'.
    
    Downloads
    ---------
        opportunities.tif: A tif file of the 'opportunities' band in ds, with colours embedded.
    """
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

    

# +
def opportunities(percent_tif, outdir='.', stub='TEST', tmpdir='.', cover_threshold=0
                  width=3, ridges=False, num_catchments=10, min_branch_length=10, 
                  contour_spacing=10, min_contour_length=100, equal_area=False):
    """
    Parameters
    ----------
        percent_tif: Percentage cover tree tif file
            - A binary tif should also work if you set the cover threshold to 0
        outdir: The output directory to save the results.
        stub: Prefix for output files. If not specified, then it appends 'categorised' to the original filename.
        cover_threshold: Percentage tree cover within a 10m pixel to be classified as a boolean 'tree'.
        width: Number of pixels away from the feature that still counts as within the buffer
            - May want different widths for different buffers later
        ridges: Whether to include opportunities for trees on the catchment boundaries (ridges)
            - If false, then the hydrolines gets used for gullies. Otherwise, the dem gets used for both gullies and ridges.
        num_catchments: Number of catchments when calculating the ridges and gullies
            - If None, then it uses the number of hydrolines in that area
        min_branch_length: Smallest allowable branch when using hydrolines to determine number of catchments
        contour_spacing: Number of pixels between each contour
            - If equal area is true, then this parameter gets used as the number of contours instead
        min_contour_length: Smallest allowable contour to use as an opportunity for planting trees
        equal_area: Whether to generate a given number of contours per tile, or just place contours on the same elevations across all tiles.
        
    Returns
    -------
        ds: an xarray with a band 'opportunities', where the integers represent the categories defined in 'opportunity_labels'.
    
    Downloads
    ---------
        opportunities.tif: A tif file of the 'opportunities' band in ds, with colours embedded.
    """
    # Load binary trees
    da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
    da_trees = da_percent > cover_threshold
    da_trees = da_trees.astype('uint8')
    ds_woody_veg = da_trees.to_dataset(name='woody_veg')

    # Load hydrolines
    if not ridges or num_catchments is None
        gdf_hydrolines, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent)
        da_hydrolines = ds_hydrolines['gullies']
        
    # Load roads
    gdf_roads, ds_roads = hydrolines(None, roads_gdb, outdir=tmpdir, stub=stub, savetif=True, save_gpkg=True, da=da_percent, layer='NationalRoads_2025_09')
    da_roads = ds_roads['gullies']  # dodgy string hardcoding of 'gullies' in hydrolines that I should fix 

    # Prepare the bboxs for cropping/stitching
    gs_bounds = gpd.GeoSeries([box(*da_trees.rio.bounds())], crs=da_trees.rio.crs)
    bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
    bbox_3857 = list(gs_bounds.to_crs('EPSG:3857').bounds.iloc[0])

    # Load worldcover 
    worldcover_stub = f'{stub}_worldcover_opportunities_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}   # Need to make unique for parallelisation
    mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, worldcover_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    da_worldcover2 = da_worldcover.rio.reproject_match(da_trees) # Should do this within full_pipelines so it doesn't need to happen twice

    # Create a cropped/stitching dem for the region of interest
    dem_stub = f'{stub}_dem_opportunities_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}   # Need to make unique for parallelisation
    mosaic, out_meta = merge_tiles_bbox(bbox_3857, tmpdir, dem_stub, nsw_dem_dir, nsw_dem_gpkg, 'filename', verbose=True) 
    ds_dem = merged_ds(mosaic, out_meta, 'dem')   # This is a 5m dem, as opposed to the 10m tree raster

    # Should reproject_match the tree_tif before saving. Then also use this reprojected version for the contours too.

    dem_tif = os.path.join(tmpdir, f'{dem_stub}_dem.tif')    
    ds_dem['dem'].astype('float64').rio.to_raster(filename)  # Needs to be float64 for catchments.py to work efficiently
    print(f"Saved: {dem_tif}")

    # Use the hydrolines to determine the relevant number of catchments
    if num_catchments is None:
        river_mask = skeletonize(da_hydrolines.values)
        branch_labels = segmentation(river_mask)
        num_segments = len(np.unique(branch_labels))
        num_catchments = num_segments * 2/3  # Each intersection in segmentation has 3 segments, whereas in catchments it has 2.    

        # Saving the branches for debugging. Will probably want to do something like this for my shelterbelts too.
        # ds_woody_veg['branch_labels'] = ('y', 'x'), branch_labels
        # ds_woody_veg['branch_labels'].astype(float).rio.to_raster('/scratch/xe2/cb8590/tmp/branch_labels_consecutive.tif')

    # Generate the gullies and ridges
    ds_catchments = catchments(dem_tif, outdir=tmpdir, stub="TEST", tmpdir=tmpdir, num_catchments=num_catchments, savetif=False, plot=False) 

    if equal_area:
        contours_array = contours_equal_area(ds_dem['dem'], num_contours=contour_spacing, min_contour_length=min_contour_length)
    else:
        contours_array = contours_interval(ds_dem['dem'], contour_spacing, min_contour_length)
        
    # Create a single xarray with all the layers

    # ds_catchments['contours'] = (["y", "x"], grid)  # Might have to reproject_match to the woody_veg at some point
    # ds_catchments['contours'].astype(float).rio.to_raster('/scratch/xe2/cb8590/tmp/contours_evenly_spaced.tif')


# -

if __name__ == '__main__':
    
    opportunities(percent_tif)


# +
# %%time
percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2024_actnsw_4326/subfolders/lat_34_lon_140/34_13-141_90_y2024_predicted.tif' # Should be fine
tmpdir = '/scratch/xe2/cb8590/'
stub='TEST'
buffer_width=3
cover_threshold=50

opportunities(percent_tif, tmpdir, stub, tmpdir, cover_threshold)
