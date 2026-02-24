# +
# Some code taken from my honours in 2021 and updated during my phd in 2024 and 2025: 
# https://gitlab.com/civilise-ai/tl-2024/-/blob/main/dem-server/Backend/ridge_and_gully/catchments.py

# +
import os
import argparse

import numpy as np
from scipy import ndimage
from skimage.morphology import thin
import rasterio
import rioxarray as rxr

from DAESIM_preprocess.topography import dirmap, pysheds_accumulation
from shelterbelts.utils.visualisation import tif_categorical
# -

import matplotlib.pyplot as plt

gullies_cmap = {
    0: (255, 255, 255),
    1: (0, 0, 255),
}
ridges_cmap = {
    0: (255, 255, 255),
    1: (255, 0, 0),
}

# +
# Monkey patch extract_river_network - their function seems to be broken in later versions of numpy because of line 1417 should be np.false_ instead of the python type False
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
    
    # ridges = edges > 0 # This gives edges that are generally 3 pixels wide
    ridges = thin(edges > 0) # This makes the edges 1 wide
    # ridges = skeletonize(edges > 0) # This is another option that looks similar to thinning

    return ridges


def plot_catchments(ds, filename=None):
    """Pretty visualisation of the terrain and ridges and gullies
    ds needs to have at least 3 bands: terrain (int or float), gullies (bool), ridges (bool)"""
    left, bottom, right, top = ds.rio.bounds()
    extent = (left, right, bottom, top)
    
    # Background
    dem = ds['terrain']
    plt.imshow(dem, cmap='terrain', interpolation='bilinear', extent=extent)
    plt.colorbar(label='height above sea level (m)')
    
    # Contours
    contour_levels = np.arange(np.floor(np.nanmin(dem)), np.ceil(np.nanmax(dem)), 10)
    contours = plt.contour(dem, levels=contour_levels, colors='black',
                           linewidths=0.5, alpha=0.5, extent=extent)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    
    # Ridges & Gullies
    transform = ds.rio.transform()
    rows, cols = np.where(ds['gullies'])
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    rows_r, cols_r = np.where(ds['ridges'])
    xr, yr = rasterio.transform.xy(transform, rows_r, cols_r)
    plt.scatter(xs, ys, marker='.', linewidths=0.01, c="blue", label='Gullies')
    plt.scatter(xr, yr, marker='.', linewidths=0.01, c="red", label='Catchments')

    plt.legend(loc='upper right', markerscale=2)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        plt.show()



def catchments(terrain_tif, outdir=".", stub="TEST", tmpdir=".", num_catchments=10, savetif=True, plot=True):
    """
    Generate gully and ridge tifs from a digital elevation model.

    Parameters
    ----------
    terrain_tif : str
        Path to the DEM (Digital Elevation Model) GeoTIFF file.
    outdir : str, optional
        Output directory for saving results. Default is current directory.
    stub : str, optional
        Prefix for output filenames. Default is "TEST".
    tmpdir : str, optional
        Temporary folder to save the terrain_tif as float64 for pysheds.
        Default is current directory.
    num_catchments : int, optional
        The number of catchments to find when assigning gullies and ridges.
        Default is 10.
    savetif : bool, optional
        Whether to save the results as a GeoTIFF. Default is True.
    plot : bool, optional
        Whether to generate a PNG visualisation. Default is True.

    Returns
    -------
    xarray.Dataset
        Dataset containing:
        
        - **terrain**: The input DEM (float)
        - **gullies**: Boolean array of gullies
        - **ridges**: Boolean array of catchment boundaries

    Notes
    -----
    When ``savetif=True``, it writes:
    
    - ``{stub}_gullies.tif``
    - ``{stub}_ridges.tif``

    When ``plot=True``, it writes:
    ``{stub}_gullies_and_ridges.png``

    """
    da = rxr.open_rasterio(terrain_tif).isel(band=0).drop_vars('band')
    
    # Make sure the input dtype is np.float64 for pysheds to work in the latest version of numpy
    if da.dtype != 'float64':
        da = da.astype(np.float64)
        terrain_tif = os.path.join(tmpdir, f"{stub}_terrain.tif")
        da.rio.to_raster(terrain_tif)
        print("Saved as float64: ", terrain_tif)
    
    grid, dem, fdir, acc = pysheds_accumulation(terrain_tif)
    gullies, full_branches = catchment_gullies(grid, fdir, acc, num_catchments) # Don't worry about the warning here
    ridges = catchment_ridges(grid, fdir, acc, full_branches)

    ds = da.to_dataset(name='terrain')
    ds['gullies'] = (["y", "x"], gullies)
    ds['ridges'] = (["y", "x"], ridges)

    if savetif:
        filename_gullies = os.path.join(outdir, f"{stub}_gullies.tif")
        filename_ridges = os.path.join(outdir, f"{stub}_ridges.tif")
        tif_categorical(ds['gullies'], filename_gullies, colormap=gullies_cmap)
        tif_categorical(ds['ridges'], filename_ridges, colormap=ridges_cmap)

    if plot:
        filename_gullies_ridges_png = os.path.join(outdir, f"{stub}_gullies_and_ridges.png")
        plot_catchments(ds, filename_gullies_ridges_png)
        
    return ds



def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('terrain_tif', help='Path to the DEM (Digital Elevation Model) GeoTIFF file')
    parser.add_argument('--outdir', default='.', help='Output directory for saving results')
    parser.add_argument('--stub', default='TEST', help='Prefix for output filenames')
    parser.add_argument('--tmpdir', default='.', help='Temporary folder to save terrain_tif as float64 for pysheds (default: current directory)')
    parser.add_argument('--num_catchments', default=10, type=int, help='The number of catchments to find (default: 10)')
    parser.add_argument('--no-save-tif', dest='savetif', action="store_false", default=True, help='Disable saving GeoTIFF output (default: enabled)')
    parser.add_argument('--no-plot', dest='plot', action="store_false", default=True, help='Disable PNG visualisation (default: enabled)')

    return parser


if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    
    catchments(args.terrain_tif, outdir=args.outdir, stub=args.stub, tmpdir=args.tmpdir, num_catchments=args.num_catchments, savetif=args.savetif, plot=args.plot)


# outdir = '../../../outdir/'
# stub = 'g2_26729'
# filename_terrain_tiles = os.path.join(outdir, f"{stub}_terrain.tif")
# filename_DEM_H = "/Users/christopherbradley/Documents/PHD/Data/DEM_Samples/Hydro_Enforced_1_Second_DEM_470734/Hydro_Enforced_1_Second_DEM.tif"
# filename_1m = "/Users/christopherbradley/Documents/PHD/Data/DEM_Samples/NSW Government - Spatial Services/DEM/1 Metre/Young201709-LID1-AHD_6306194_55_0002_0002_1m.tif"
# filename_5m = "/Users/christopherbradley/Documents/PHD/Data/DEM_Samples/NSW Government - Spatial Services/DEM/5 Metre/Young201702-PHO3-AHD_6306194_55_0002_0002_5m.tif"
# filename_DEM_S = "/Users/christopherbradley/Documents/PHD/Data/DEM_Samples/1_Second_DEM_Smoothed_470806/1_Second_DEM_Smoothed.tif"
# filename_DEM_Normal = "/Users/christopherbradley/Documents/PHD/Data/DEM_Samples/1_Second_DEM_470805/1_Second_DEM.tif"

# # ds = catchments(filename_1m, outdir="../../../outdir", stub="g2_26729_1m")
# # ds = catchments(filename_5m, outdir="../../../outdir", stub="g2_26729_5m")
# ds = catchments(filename_DEM_Normal, outdir="../../../outdir", stub="g2_26729_DEM-Normal")
# plot_catchments(ds)
