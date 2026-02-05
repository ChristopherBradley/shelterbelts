# +
import os
import glob
import pathlib

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

from shelterbelts.utils.visualization import tif_categorical, visualise_categories
from shelterbelts.utils.crop_and_rasterize import crop_and_rasterize
from shelterbelts.apis.canopy_height import merge_tiles_bbox, merged_ds
from shelterbelts.apis.catchments import catchments  # This takes a while to import
from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.utils.filepaths import nsw_dem_dir

# -

# from shelterbelts.indices.full_pipelines import worldcover_dir, worldcover_geojson, hydrolines_gdb, roads_gdb

# +
nsw_dem_gpkg = 'cb8590_NSW_5m_DEMs_3857_footprints.gpkg'

opportunity_cmap = {  # Should refactor to make this plural for consistency
    0:(255, 255, 255),
    5:(29, 153, 105),
    6:(127, 168, 57),
    7:(129, 146, 124),
    8:(190, 160, 60)
}
opportunity_labels = {
    0:'',
    5:'Opportunities in Gullies',
    6:'Opportunities on Ridges',
    7:'Opportunities next to Roads',
    8:'Opportunities along Contours'
}
inverted_labels = {v: k for k, v in opportunity_labels.items()}


# +
def segmentation(river_mask, min_branch_length=10):
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
    minZ, maxZ = np.nanmin(Z), np.nanmax(Z)

    # contours = [height for height in range(minZ, maxZ) if height % interval == 0]
    contours = [height for height in range(int(minZ), int(maxZ)) if height % interval == 0]
    
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


# -

def opportunities_da(da_trees, da_roads, da_gullies, da_ridges, da_contours, da_worldcover, outdir='.', stub='TEST', tmpdir='.', 
                     width=3, savetif=True, plot=False, crop_pixels=0):
    """Suggest opportunities for new trees based on gullies, roads, ridges and contours - in that order of priority (bit arbitrary).
    
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
    grass_crops = (da_worldcover == 30) | (da_worldcover == 40)  # Either grass pixels or crop pixels
    
    # Setup the buffer kernel
    y, x = np.ogrid[-width:width+1, -width:width+1]
    gap_kernel = (x**2 + y**2 <= width**2)
    
    # Buffered gullies
    buffered_gullies = binary_dilation(da_gullies.values, structure=gap_kernel)
    gully_opportunities = buffered_gullies & grass_crops & ~da_trees
    
    # Buffered roads
    buffered_roads = binary_dilation(da_roads.values, structure=gap_kernel)
    road_opportunities = buffered_roads & grass_crops & ~da_trees & ~gully_opportunities
    
    # Buffered ridges
    if da_ridges is None:
        buffered_ridges = ridge_opportunities = np.zeros_like(da_trees, dtype=np.uint8)
    else:
        buffered_ridges = binary_dilation(da_ridges.values, structure=gap_kernel)
        ridge_opportunities = buffered_ridges & grass_crops & ~da_trees & ~gully_opportunities & ~buffered_roads
    
    # Buffered contours
    buffered_contours = binary_dilation(da_contours.values, structure=gap_kernel)
    contour_opportunities = buffered_contours & grass_crops & ~da_trees & ~gully_opportunities & ~buffered_roads & ~buffered_ridges  # Could just override a single numpy array as we go.
    
    # There should be no overlaps, so the order of assigning shouldn't matter
    opportunities = np.zeros_like(da_trees, dtype=np.uint8)
    opportunities[gully_opportunities.astype(bool)] = 5   
    opportunities[ridge_opportunities.astype(bool)] = 6
    opportunities[road_opportunities.astype(bool)] = 7
    opportunities[contour_opportunities.astype(bool)] = 8
    
    # Creating the xarray
    ds = da_trees.to_dataset(name='woody_veg')
    ds['opportunities'] = ('y', 'x'), opportunities

    # Crop the output if it was expanded before the pipeline started
    if crop_pixels is not None and crop_pixels != 0:
        ds = ds.isel(
            x=slice(crop_pixels, -crop_pixels),
            y=slice(crop_pixels, -crop_pixels)
        )
    
    if savetif:
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir,f"{stub}_opportunities.tif")
        tif_categorical(ds['opportunities'], filename, opportunity_cmap)

    if plot:
        # filename_png = os.path.join(outdir, f"{stub}_opportunities.png")
        # visualise_categories(ds['opportunities'], filename_png, opportunity_cmap, opportunity_labels, "Opportunities")
        visualise_categories(ds['opportunities'], None, opportunity_cmap, opportunity_labels, "Opportunities")

    return ds


def opportunities(percent_tif, outdir='.', stub=None, tmpdir='.', cover_threshold=0,
                  width=3, ridges=False, num_catchments=10, min_branch_length=10, 
                  contour_spacing=10, min_contour_length=100, equal_area=False, 
                  savetif=True, plot=False, crop_pixels=0):
    """Suggest opportunities for new trees based on ridges, gullies and contours.
    
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

    if stub is None:
        tif_stem = pathlib.Path(percent_tif).stem
        stub = f'{tif_stem}_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}'   # Need to make unique for parallelisation
        
    # Load binary trees
    da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
    da_trees = da_percent > cover_threshold
    da_trees = da_trees.astype('uint8')
    ds = da_trees.to_dataset(name='woody_veg')

    # Load hydrolines
    if not ridges or num_catchments is None:
        gdf_hydrolines, ds_hydrolines = hydrolines(None, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, da=da_percent)
        da_hydrolines = ds_hydrolines['gullies']
        
    # Load roads
    gdf_roads, ds_roads = hydrolines(None, roads_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, da=da_percent, layer='NationalRoads_2025_09')
    da_roads = ds_roads['gullies']  # dodgy string hardcoding of 'gullies' in hydrolines that I should fix 

    # Prepare the bboxs for cropping/stitching
    gs_bounds = gpd.GeoSeries([box(*da_trees.rio.bounds())], crs=da_trees.rio.crs)
    bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
    bbox_3857 = list(gs_bounds.to_crs('EPSG:3857').bounds.iloc[0])

    # Load worldcover 
    unique_stub = stub
    mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, unique_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
    ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
    da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
    da_worldcover2 = da_worldcover.rio.reproject_match(da_trees) # Should do this within full_pipelines so it doesn't need to happen twice

    # Create a cropped/stitching dem for the region of interest
    dem_stub = f'{stub}_dem_opportunities_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}'   # Need to make unique for parallelisation
    
    # Need to check if the region is inside NSW, and use the Australia version or terrain tiles if not.
    mosaic, out_meta = merge_tiles_bbox(bbox_3857, tmpdir, unique_stub, nsw_dem_dir, nsw_dem_gpkg, 'filename', verbose=False) 
    ds_dem = merged_ds(mosaic, out_meta, 'dem')   # This is a 5m dem, as opposed to the 10m tree raster
    ds_dem = ds_dem.rio.reproject_match(ds)

    dem_tif = os.path.join(tmpdir, f'{dem_stub}_dem.tif')    
    ds_dem['dem'].astype('float64').rio.to_raster(dem_tif)  # Needs to be float64 for catchments.py to work efficiently
    # print(f"Saved: {dem_tif}")

    # Use the hydrolines to determine the relevant number of catchments
    if num_catchments is None:
        river_mask = skeletonize(da_hydrolines.values)
        branch_labels = segmentation(river_mask)
        num_segments = len(np.unique(branch_labels))
        num_catchments = num_segments * 2/3  # Each intersection in segmentation has 3 segments, whereas in catchments it has 2.    

        # Saving the branches for debugging. Will probably want to do something like this for my shelterbelts too.
        # ds['branch_labels'] = ('y', 'x'), branch_labels
        # ds['branch_labels'].astype(float).rio.to_raster('/scratch/xe2/cb8590/tmp/branch_labels_consecutive.tif')

    # Generate the gullies and ridges. 
    ds_catchments = catchments(dem_tif, outdir=tmpdir, stub="TEST", tmpdir=tmpdir, num_catchments=num_catchments, savetif=False, plot=False) 
    
    # Might want to remove really small gullies and ridges like I do with the contours.

    # Generate the contours
    if equal_area:
        contours_array = contours_equal_area(ds_dem['dem'], num_contours=contour_spacing, min_contour_length=min_contour_length)
    else:
        contours_array = contours_interval(ds_dem['dem'], contour_spacing, min_contour_length)
        
    # Create a single xarray with all the layers
    ds['roads'] = da_roads
    if ridges:
        ds['gullies'] = ds_catchments['gullies']
        ds['ridges'] = ds_catchments['ridges']
        da_ridges = ds['ridges']
    else:
        ds['gullies'] = da_hydrolines
        da_ridges = None
    ds['contours'] = (["y", "x"], contours_array)  
    ds['worldcover'] = da_worldcover2 

    ds_opportunities = opportunities_da(ds['woody_veg'], ds['roads'], ds['gullies'], da_ridges, ds['contours'], ds['worldcover'],
                                       outdir, unique_stub, tmpdir, width, savetif, plot, crop_pixels)
    
    return ds_opportunities


# Could generalise and reuse the run_pipeline_tifs function from full_pipelines.py. The only issue is that would mean the parameters would have to be passed as *args or **kwargs which I think is less readable.
def opportunities_folder(folder, stub=None, tmpdir='.', cover_threshold=0,
                  width=3, ridges=False, num_catchments=10, min_branch_length=10, 
                  contour_spacing=10, min_contour_length=100, equal_area=False, 
                  savetif=True, plot=False, crop_pixels=0, limit=None):
    """Run the opportunities function on every tif in the folder and mosaic the outputs at the end"""

    folder_stem = pathlib.Path(folder).stem
    if not stub:
        stub = f'opportunities_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}'   # Need to make unique for parallelisation
    
    outdir = os.path.join(folder, f'{folder_stem}_{stub}')
    os.makedirs(outdir, exist_ok=True)
    percent_tifs = glob.glob(f'{folder}/*.tif')
    if limit:
        percent_tifs = percent_tifs[:limit]
    for percent_tif in percent_tifs:
        tif_stem = pathlib.Path(percent_tif).stem
        tif_stub = f"{tif_stem}_{stub}"     # This stub needs to include the exact lat lon of the tile. 
        opportunities(percent_tif, outdir, tif_stub, tmpdir, cover_threshold,
                  width, ridges, num_catchments, min_branch_length, 
                  contour_spacing, min_contour_length, equal_area, 
                  savetif, plot, crop_pixels)
        
    # outdir = folder
    # Could just use the merge_lidar function instead of reimplementing like this each time.
    gdf = bounding_boxes(outdir, filetype='opportunities.tif', stub=stub)
    
    footprint_gpkg = f"{stub}_footprints.gpkg"
    bbox =[gdf.bounds['minx'].min(), gdf.bounds['miny'].min(), gdf.bounds['maxx'].max(), gdf.bounds['maxy'].max()]
    mosaic, out_meta = merge_tiles_bbox(bbox, tmpdir, "", outdir, footprint_gpkg, id_column='filename')  # The tilenames should already be unique so we don't need a stub
    ds = merged_ds(mosaic, out_meta, 'opportunities')

    basedir = os.path.dirname(outdir)
    filename_linear = os.path.join(basedir, f'{stub}_merged.tif')
    tif_categorical(ds['opportunities'], filename_linear, opportunity_cmap) 
    return ds


# +
# Would be slightly more computationally efficient to generate the opportunities from within full_pipelines than from it's own pbs script, since we already load a bunch of the layers (trees, hydrolines, worldcover)
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Suggest opportunities for new trees based on ridges, gullies, and contours."
    )

    parser.add_argument("percent_tif", help="Input percentage cover tree tif file")
    parser.add_argument("--outdir", default=".", help="Output directory for results (default: current directory)")
    parser.add_argument("--stub", default=None, help="Prefix for output files (default: None)")
    parser.add_argument("--tmpdir", default=".", help="Temporary working directory (default: current directory)")
    parser.add_argument("--cover_threshold", type=int, default=0, help="Tree cover threshold percentage (default: 0)")
    parser.add_argument("--width", type=int, default=3, help="Buffer width in pixels (default: 3)")
    parser.add_argument("--ridges", action="store_true", help="Include opportunities on ridges (default: False)")
    parser.add_argument("--num_catchments", type=int, default=10, help="Number of catchments for ridges/gullies (default: 10)")
    parser.add_argument("--min_branch_length", type=int, default=10, help="Minimum branch length for hydrolines (default: 10)")
    parser.add_argument("--contour_spacing", type=int, default=10, help="Pixel spacing between contours (default: 10)")
    parser.add_argument("--min_contour_length", type=int, default=100, help="Minimum contour length to consider (default: 100)")
    parser.add_argument("--equal_area", action="store_true", help="Use equal-area contours instead of fixed elevation spacing (default: False)")
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots (default: False)")
    parser.add_argument("--crop_pixels", type=int, default=0, help="Number of pixels to crop from the linear_tif (default: 0)")
    parser.add_argument("--limit", type=int, default=None, help="Number of tifs to process (default: all)")

    return parser.parse_args()


# -


if __name__ == "__main__":
    args = parse_arguments()

    if args.percent_tif.endswith('.tif'):
        opportunities(
            percent_tif=args.percent_tif,
            outdir=args.outdir,
            stub=args.stub,
            tmpdir=args.tmpdir,
            cover_threshold=args.cover_threshold,
            width=args.width,
            ridges=args.ridges,
            num_catchments=args.num_catchments,
            min_branch_length=args.min_branch_length,
            contour_spacing=args.contour_spacing,
            min_contour_length=args.min_contour_length,
            equal_area=args.equal_area,
            savetif=True,
            plot=args.plot,
            crop_pixels=args.crop_pixels
        )
    else:
        opportunities_folder(
            folder=args.percent_tif,
            stub=args.stub,
            tmpdir=args.tmpdir,
            cover_threshold=args.cover_threshold,
            width=args.width,
            ridges=args.ridges,
            num_catchments=args.num_catchments,
            min_branch_length=args.min_branch_length,
            contour_spacing=args.contour_spacing,
            min_contour_length=args.min_contour_length,
            equal_area=args.equal_area,
            savetif=True,
            plot=args.plot,
            crop_pixels=args.crop_pixels,
            limit=args.limit
        )

