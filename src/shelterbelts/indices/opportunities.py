# +
import os
import glob
import pathlib
import argparse

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box      
from scipy.ndimage import binary_dilation

import networkx as nx
from collections import Counter
from skimage.morphology import skeletonize
from skimage.measure import find_contours


from shelterbelts.utils.visualisation import tif_categorical, visualise_categories
from shelterbelts.utils.tiles import merge_tiles_bbox, merged_ds, crop_and_rasterize
from shelterbelts.indices.catchments import catchments  # This takes a while to import
from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.utils.filepaths import nsw_dem_dir, hydrolines_gdb, roads_gdb

# -

# from shelterbelts.indices.all_indices import worldcover_dir, worldcover_geojson, hydrolines_gdb, roads_gdb

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
        # Check 8-connectivity neighbours
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < river_mask.shape[0] and 0 <= nx_ < river_mask.shape[1]:
                    if river_mask[ny, nx_] == 1:
                        G.add_edge((y, x), (ny, nx_))
    
    # Identify junctions (>=3 neighbours) and endpoints (==1 neighbour)
    junctions = {n for n, d in G.degree() if d >= 3}
    endpoints = {n for n, d in G.degree() if d == 1}
    
    # Extract branches: paths between junctions and endpoints
    branches = []
    visited_edges = set()
    
    for start in endpoints | junctions:
        for neighbour in G.neighbors(start):
            edge = frozenset([start, neighbour])
            if edge in visited_edges:
                continue
    
            branch = [start, neighbour]
            visited_edges.add(edge)
            prev, current = start, neighbour
    
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
    
            # Find neighbouring branches
            dilated = binary_dilation(mask, structure=np.ones((3, 3)))
            neighbours = np.unique(branch_labels_new[dilated & (branch_labels_new != bid) & (branch_labels_new != 0)])
    
            if len(neighbours) == 0:
                continue
    
            # Merge into largest neighbouring branch
            largest_neighbour = max(neighbours, key=lambda n: branch_sizes.get(n, 0))
            branch_labels_new[mask] = largest_neighbour
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

def opportunities_da(da_trees, da_roads, da_gullies, da_ridges, da_dem, da_worldcover, outdir='.', stub='TEST', tmpdir='.', 
                     width=1, contour_spacing=10, min_contour_length=100, equal_area=False,
                     savetif=True, plot=False, crop_pixels=0):
    """Classify grass/crop pixels near gullies, roads, ridges and contours as planting opportunities."""
    grass_crops = (da_worldcover == 30) | (da_worldcover == 40)  # Either grass pixels or crop pixels
    
    # Generate contours from DEM
    if not contour_spacing:
        contours_array = np.zeros_like(da_trees, dtype=np.uint8)
    elif equal_area:
        contours_array = contours_equal_area(da_dem, num_contours=contour_spacing + 1, min_contour_length=min_contour_length)
    else:
        contours_array = contours_interval(da_dem, contour_spacing, min_contour_length)
    
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
    buffered_contours = binary_dilation(contours_array, structure=gap_kernel)
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


def opportunities(percent_tif, roads_data=None, gullies_data=None, ridges_data=None, worldcover_data=None, dem_data=None,
                  outdir='.', stub=None, tmpdir='.', cover_threshold=1,
                  width=1, ridges=False, num_catchments=10, min_branch_length=10, 
                  contour_spacing=10, min_contour_length=100, equal_area=False, 
                  savetif=True, plot=False, crop_pixels=0):
    """
    Suggest opportunities for new trees based on ridges, gullies and contours.

    Loads a percent-cover GeoTIFF, derives gullies (from hydrolines or DEM
    catchments), roads, ridges and contours, then delegates to
    :func:`opportunities_da` to classify grass/crop pixels near those
    features as planting opportunities.

    Parameters
    ----------
    percent_tif : str
        Path to a percent-cover GeoTIFF. A binary tif also works with the
        default ``cover_threshold`` of 1.
    roads_data : str or xarray.DataArray, optional
        Pre-loaded binary roads raster or path to a GeoTIFF. When None the
        roads are derived from the bounding box of ``percent_tif``.
    gullies_data : str or xarray.DataArray, optional
        Pre-loaded binary gullies raster or path to a GeoTIFF. When None
        the gullies are derived from hydrolines or DEM catchments.
    ridges_data : str or xarray.DataArray, optional
        Pre-loaded binary ridges raster or path to a GeoTIFF. When None
        and ``ridges=True``, ridges are derived from DEM catchments.
    worldcover_data : str or xarray.DataArray, optional
        Pre-loaded WorldCover land-cover raster or path to a GeoTIFF. When
        None the WorldCover data is derived from the bounding box.
    dem_data : str or xarray.DataArray, optional
        Pre-loaded DEM raster or path to a GeoTIFF. When None the DEM is
        derived from the bounding box.
    outdir : str, optional
        Output directory for saving results. Default is current directory.
    stub : str, optional
        Prefix for output filenames. If not provided it is derived from
        ``percent_tif``.
    tmpdir : str, optional
        Directory for temporary files. Default is current directory.
    cover_threshold : int, optional
        Pixel percent cover threshold to treat a pixel as 'tree'. Default is 1.
    width : int, optional
        Number of pixels away from the feature that still counts as within
        the buffer. Default is 1.
    ridges : bool, optional
        Whether to include opportunities for trees on catchment boundaries
        (ridges). If False, hydrolines are used for gullies only. If True,
        the DEM is used for both gullies and ridges. Default is False.
    num_catchments : int, optional
        Number of catchments when calculating ridges and gullies. If None,
        the number of hydroline segments is used instead. Default is 10.
    min_branch_length : int, optional
        Smallest allowable branch when using hydrolines to determine number
        of catchments. Default is 10.
    contour_spacing : int, optional
        Number of pixels between each contour. If ``equal_area`` is True,
        this is used as the number of contours instead. Set to 0 to
        disable contour opportunities. Default is 10.
    min_contour_length : int, optional
        Smallest allowable contour to use as an opportunity for planting
        trees. Default is 100.
    equal_area : bool, optional
        Whether to generate a given number of contours per tile (True), or
        place contours at the same elevations across all tiles (False).
        Default is False.
    savetif : bool, optional
        Whether to save the results as a GeoTIFF. Default is True.
    plot : bool, optional
        Whether to generate a visualisation. Default is False.
    crop_pixels : int, optional
        Number of pixels to crop from each edge of the output. Default is 0.

    Returns
    -------
    xarray.Dataset
        Dataset containing:

        - **woody_veg**: Original binary tree/no-tree classification
        - **opportunities**: Opportunity categories (values 0, 5, 6, 7, 8)

    Notes
    -----
    When savetif=True, it outputs a GeoTIFF file with embedded color map:
    ``{stub}_opportunities.tif``

    Examples
    --------
    Using file paths as input:

    >>> from shelterbelts.utils.filepaths import get_filename
    >>> tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
    >>> roads_file = get_filename('g2_26729_roads.tif')
    >>> gullies_file = get_filename('g2_26729_hydrolines.tif')
    >>> dem_file = get_filename('g2_26729_DEM-H.tif')
    >>> worldcover_file = get_filename('g2_26729_worldcover.tif')
    >>> ds = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, dem_data=dem_file, worldcover_data=worldcover_file, outdir='/tmp', plot=False, savetif=False)
    >>> set(ds.data_vars) == {'woody_veg', 'opportunities'}
    True

    .. plot::

        import rioxarray as rxr
        from shelterbelts.indices.opportunities import opportunities, opportunity_cmap, opportunity_labels
        from shelterbelts.utils.filepaths import get_filename
        from shelterbelts.utils.visualisation import visualise_categories_sidebyside

        tree_file = get_filename('g2_26729_binary_tree_cover_10m.tiff')
        roads_file = get_filename('g2_26729_roads.tif')
        gullies_file = get_filename('g2_26729_hydrolines.tif')
        dem_file = get_filename('g2_26729_DEM-H.tif')
        worldcover_file = get_filename('g2_26729_worldcover.tif')
        common = dict(dem_data=dem_file, worldcover_data=worldcover_file, outdir='/tmp', plot=False, savetif=False)

        da_zero = rxr.open_rasterio(roads_file).isel(band=0).drop_vars('band') * 0

        ds_roads = opportunities(tree_file, roads_data=roads_file, gullies_data=da_zero, **common, contour_spacing=0)
        ds_gullies = opportunities(tree_file, roads_data=da_zero, gullies_data=gullies_file, **common, contour_spacing=0)
        visualise_categories_sidebyside(
            ds_roads['opportunities'], ds_gullies['opportunities'],
            colormap=opportunity_cmap, labels=opportunity_labels,
            title1="Just roads", title2="Just gullies"
        )

        ds_w1 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, width=1)
        ds_w5 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, width=5)
        visualise_categories_sidebyside(
            ds_w1['opportunities'], ds_w5['opportunities'],
            colormap=opportunity_cmap, labels=opportunity_labels,
            title1="width=1", title2="width=5"
        )

        ds_cs5 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, contour_spacing=5)
        ds_cs20 = opportunities(tree_file, roads_data=roads_file, gullies_data=gullies_file, **common, contour_spacing=20)
        visualise_categories_sidebyside(
            ds_cs5['opportunities'], ds_cs20['opportunities'],
            colormap=opportunity_cmap, labels=opportunity_labels,
            title1="contour_spacing=5", title2="contour_spacing=20"
        )
    """

    if stub is None:
        tif_stem = pathlib.Path(percent_tif).stem
        stub = f'{tif_stem}_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}'   # Need to make unique for parallelisation
        
    # Load binary trees
    da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
    da_trees = da_percent >= cover_threshold
    da_trees = da_trees.astype('uint8')
    ds = da_trees.to_dataset(name='woody_veg')

    # Load roads
    if isinstance(roads_data, xr.DataArray):
        da_roads = roads_data
    elif isinstance(roads_data, str):
        da_roads = rxr.open_rasterio(roads_data).isel(band=0).drop_vars('band')
    else:
        gdf_roads, ds_roads = crop_and_rasterize(da_percent, roads_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, layer='NationalRoads_2025_09', feature_name='roads')
        da_roads = ds_roads['roads']

    # Load gullies
    if isinstance(gullies_data, xr.DataArray):
        da_gullies = gullies_data
    elif isinstance(gullies_data, str):
        da_gullies = rxr.open_rasterio(gullies_data).isel(band=0).drop_vars('band')
    else:
        if not ridges or num_catchments is None:
            gdf_hydrolines, ds_hydrolines = crop_and_rasterize(da_percent, hydrolines_gdb, outdir=tmpdir, stub=stub, savetif=False, save_gpkg=False, feature_name='gullies')
            da_gullies = ds_hydrolines['gullies']

    # Load ridges
    if isinstance(ridges_data, xr.DataArray):
        da_ridges = ridges_data
    elif isinstance(ridges_data, str):
        da_ridges = rxr.open_rasterio(ridges_data).isel(band=0).drop_vars('band')
    else:
        da_ridges = None

    # Load worldcover
    if isinstance(worldcover_data, xr.DataArray):
        da_worldcover = worldcover_data
    elif isinstance(worldcover_data, str):
        da_worldcover = rxr.open_rasterio(worldcover_data).isel(band=0).drop_vars('band')
        da_worldcover = da_worldcover.rio.reproject_match(da_trees)
    else:
        # Prepare the bboxs for cropping/stitching
        gs_bounds = gpd.GeoSeries([box(*da_trees.rio.bounds())], crs=da_trees.rio.crs)
        bbox_4326 = list(gs_bounds.to_crs('EPSG:4326').bounds.iloc[0])
        unique_stub = stub
        mosaic, out_meta = merge_tiles_bbox(bbox_4326, tmpdir, unique_stub, worldcover_dir, worldcover_geojson, 'filename', verbose=False) 
        ds_worldcover = merged_ds(mosaic, out_meta, 'worldcover')
        da_worldcover = ds_worldcover['worldcover'].rename({'longitude':'x', 'latitude':'y'})
        da_worldcover = da_worldcover.rio.reproject_match(da_trees)

    # Load DEM
    if isinstance(dem_data, xr.DataArray):
        da_dem = dem_data
    elif isinstance(dem_data, str):
        da_dem = rxr.open_rasterio(dem_data).isel(band=0).drop_vars('band')
    else:
        gs_bounds = gpd.GeoSeries([box(*da_trees.rio.bounds())], crs=da_trees.rio.crs)
        bbox_3857 = list(gs_bounds.to_crs('EPSG:3857').bounds.iloc[0])
        unique_stub = stub
        dem_stub = f'{stub}_dem_opportunities_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}'
        mosaic, out_meta = merge_tiles_bbox(bbox_3857, tmpdir, unique_stub, nsw_dem_dir, nsw_dem_gpkg, 'filename', verbose=False) 
        ds_dem = merged_ds(mosaic, out_meta, 'dem')
        ds_dem = ds_dem.rio.reproject_match(ds)
        da_dem = ds_dem['dem']

    # If gullies/ridges not provided and not using pre-loaded data, derive from catchments
    if gullies_data is None and ridges_data is None and ridges:
        dem_stub = f'{stub}_dem_opportunities_w{width}_r{ridges}_nc{num_catchments}_bl{min_branch_length}_cs{contour_spacing}_cl{min_contour_length}_e{equal_area}'
        dem_tif = os.path.join(tmpdir, f'{dem_stub}_dem.tif')
        da_dem.astype('float64').rio.to_raster(dem_tif)

        if num_catchments is None:
            river_mask = skeletonize(da_gullies.values)
            branch_labels = segmentation(river_mask)
            num_segments = len(np.unique(branch_labels))
            num_catchments = num_segments * 2/3

        ds_catchments = catchments(dem_tif, outdir=tmpdir, stub="TEST", tmpdir=tmpdir, num_catchments=num_catchments, savetif=False, plot=False)
        da_gullies = ds_catchments['gullies']
        da_ridges = ds_catchments['ridges']

    ds_opportunities = opportunities_da(ds['woody_veg'], da_roads, da_gullies, da_ridges, da_dem, da_worldcover,
                                       outdir, stub, tmpdir, width, contour_spacing, min_contour_length, equal_area,
                                       savetif, plot, crop_pixels)
    
    return ds_opportunities


# Could generalise and reuse the indices_tifs function from all_indices.py (previously run_pipeline_tifs/full_pipelines.py). The only issue is that would mean the parameters would have to be passed as *args or **kwargs which I think is less readable.
def opportunities_folder(folder, stub=None, tmpdir='.', cover_threshold=1,
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
        opportunities(percent_tif, outdir=outdir, stub=tif_stub, tmpdir=tmpdir,
                  cover_threshold=cover_threshold, width=width, ridges=ridges,
                  num_catchments=num_catchments, min_branch_length=min_branch_length,
                  contour_spacing=contour_spacing, min_contour_length=min_contour_length,
                  equal_area=equal_area, savetif=savetif, plot=plot, crop_pixels=crop_pixels)
        
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
# Would be slightly more computationally efficient to generate the opportunities from within all_indices.py than from it's own pbs script, since we already load a bunch of the layers (trees, hydrolines, worldcover)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Suggest opportunities for new trees based on ridges, gullies, and contours."
    )

    parser.add_argument("percent_tif", help="Input percentage cover tree tif file")
    parser.add_argument("--outdir", default=".", help="Output directory for saving results")
    parser.add_argument("--stub", default=None, help="Prefix for output filenames")
    parser.add_argument("--tmpdir", default=".", help="Temporary working directory (default: current directory)")
    parser.add_argument("--cover_threshold", type=int, default=1, help="Tree cover threshold percentage (default: 1)")
    parser.add_argument("--width", type=int, default=1, help="Buffer width in pixels (default: 1)")
    parser.add_argument("--ridges", action="store_true", help="Include opportunities on ridges (default: False)")
    parser.add_argument("--num_catchments", type=int, default=10, help="Number of catchments for ridges/gullies (default: 10)")
    parser.add_argument("--min_branch_length", type=int, default=10, help="Minimum branch length for hydrolines (default: 10)")
    parser.add_argument("--contour_spacing", type=int, default=10, help="Pixel spacing between contours (default: 10)")
    parser.add_argument("--min_contour_length", type=int, default=100, help="Minimum contour length to consider (default: 100)")
    parser.add_argument("--equal_area", action="store_true", help="Use equal-area contours instead of fixed elevation spacing (default: False)")
    parser.add_argument("--no-savetif", dest="savetif", action="store_false", default=True, help="Disable saving GeoTIFF output (default: enabled)")
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots (default: False)")
    parser.add_argument("--crop_pixels", type=int, default=0, help="Number of pixels to crop from each edge of the output")
    parser.add_argument("--limit", type=int, default=None, help="Number of tifs to process (default: all)")

    return parser


# -


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()

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
            savetif=args.savetif,
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
            savetif=args.savetif,
            plot=args.plot,
            crop_pixels=args.crop_pixels,
            limit=args.limit
        )

