
import os
import argparse

import numpy as np
import rioxarray as rxr
from scipy.ndimage import label, binary_erosion, binary_dilation

from shelterbelts.apis.worldcover import tif_categorical, visualise_categories

# Create a single array with all the layers
tree_categories_cmap = {
    0:(255, 255, 255),
    11:(122, 82, 0),
    12:(8, 79, 0),
    13:(14, 138, 0),
    14:(22, 212, 0)
}
tree_categories_labels = {
    11:'Scattered Trees',
    12:'Patch Core',
    13:'Patch Edge',
    14:'Corridor (other)',
    0:'Not trees'
}
inverted_labels = {v: k for k, v in tree_categories_labels.items()}


def tree_clusters(woody_veg, max_gap_size=2):
    """"Assign a cluster label to trees within a given distance of each other.
        Using a gap_size of 0 means only direct adjacencies using the 4 neighbour rule."""

    # Create the circular kernel
    y, x = np.ogrid[-max_gap_size:max_gap_size+1, -max_gap_size:max_gap_size+1]
    gap_kernel = (x**2 + y**2 <= max_gap_size**2)
    
    # Dilate the woody veg based on this kernel, to connect tree clusters that are close together
    dilated = binary_dilation(woody_veg, structure=gap_kernel)
    
    # Label the clusters
    labeled_dilated, num_features = label(dilated)
    
    # Remove the dilations
    trees_labelled = labeled_dilated * woody_veg
    
    return trees_labelled


def scattered_trees(trees_labelled, min_patch_size=20):
    """Create a boolean mask of small patches based on the labelled clusters."""
    counts = np.bincount(trees_labelled.ravel())
    small_clusters = np.flatnonzero(counts < min_patch_size)
    scattered_area = np.isin(trees_labelled, small_clusters)
    return scattered_area


def core_trees(woody_veg, edge_size=3, min_patch_size=20):
    """Find pixels surrounded by a circle of trees in every direction with radius edge_size"""    
    y, x = np.ogrid[-edge_size:edge_size+1, -edge_size:edge_size+1]
    core_kernel = (x**2 + y**2 <= edge_size**2)
    all_core_area = binary_erosion(woody_veg, structure=core_kernel)
    
    # Exclude core area's smaller than the scattered trees threshold
    labeled_cores, num_features = label(all_core_area)
    large_cores = np.flatnonzero(np.bincount(labeled_cores.ravel()) >= min_patch_size)[1:] # Drop the 0 category
    core_area = np.isin(labeled_cores, large_cores)

    return core_area, core_kernel


def tree_categories(filename, outdir='.', stub=None, min_patch_size=20, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True):
    """Categorise a boolean woody veg tif into scattered trees, edges, core areas, and corridors, based on the Fragstats landscape ecology approach

    Parameters
    ----------
        filename: A binary tif file containing tree/no tree information.
        outdir: The output directory to save the results.
        stub: Prefix for output files. If not specified, then it appends 'categorised' to the original filename.
        min_patch_size: The minimum area to be classified as a patch/corrider rather than just scattered trees.
        edge_size: The buffer distance at the edge of a patch, with pixels inside this being the 'core area'. 
            Non-scattered tree pixels outside the core area but within edge_size pixels of a core area get defined as 'edge' pixels. Otherwise they become 'corridor' pixels.
        max_gap_size: The allowable gap between two tree clusters before considering them separate patches.
        ds: a pre-loaded xarray.DataSet with a band 'woody_veg'. This gets used instead of the woody_veg_tif when provided.
        save_tif: Boolean to determine whether to save the final result to file.
        plot: Boolean to determine whether to generate a png plot.

    Returns
    -------
        ds: an xarray with a band 'tree_categories', where the integers represent the categories defined in 'tree_category_labels'.

    Downloads
    ---------
        woody_veg_categorised.tif: A tif file of the 'tree_categories' band in ds, with colours embedded.
        woody_veg_categorised.png: A png file like the tif file, but with a legend as well.
    
    """
    da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
    ds = da.to_dataset(name='woody_veg')
    woody_veg = ds['woody_veg'].values.astype(bool)

    trees_labelled = tree_clusters(woody_veg, max_gap_size)
    scattered_area = scattered_trees(trees_labelled, min_patch_size)
    core_area, core_kernel = core_trees(woody_veg, edge_size, min_patch_size)
    edge_area = binary_dilation(core_area, structure=core_kernel) & ~core_area & woody_veg
    corridor_area = woody_veg & ~(core_area | edge_area | scattered_area)

    tree_categories = np.zeros_like(woody_veg, dtype=np.uint8)
    tree_categories[scattered_area] = inverted_labels['Scattered Trees']
    tree_categories[core_area]      = inverted_labels['Patch Core']
    tree_categories[edge_area]      = inverted_labels['Patch Edge']
    tree_categories[corridor_area]  = inverted_labels['Corridor (other)']
    ds['tree_categories'] = (('y', 'x'), tree_categories)
    ds = ds.rename({'x':'longitude', 'y': 'latitude'})

    if not stub:
        # Use the same prefix as the original woody_veg filename
        stub = filename.split('/')[-1].split('.')[0]

    if save_tif:
        filename_categorical = os.path.join(outdir,f"{stub}_categorised.tif")
        tif_categorical(ds['tree_categories'], filename_categorical, tree_categories_cmap)

    if plot:
        filename_categorical_png = os.path.join(outdir, f"{stub}_categorised.png")
        visualise_categories(ds['tree_categories'], filename_categorical_png, tree_categories_cmap, tree_categories_labels, "Tree Categories")
                
    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', help='A binary tif file containing tree/no tree information')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', default=None, help='Prefix for output files.')
    parser.add_argument('--min_patch_size', default=20, help='The minimum area to be classified as a patch/corrider rather than just scattered trees.')
    parser.add_argument('--edge_size', default=3, help='The buffer distance at the edge of a patch, with pixels inside this being the core area')
    parser.add_argument('--max_gap_size', default=2, help='The allowable gap between two tree clusters before considering them as separate patches.')
    parser.add_argument('--plot', default=False, action="store_true", help="Boolean to Save a png file along with the tif")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    
    filename = args.filename
    outdir = args.outdir
    stub = args.stub
    min_patch_size = int(args.min_patch_size)
    edge_size = int(args.edge_size)
    max_gap_size = int(args.max_gap_size)
    plot = args.plot
    
    tree_categories(filename, outdir, stub, min_patch_size, edge_size, max_gap_size, plot=plot)

