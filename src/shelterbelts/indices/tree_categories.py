import os
import argparse

import numpy as np
import rioxarray as rxr
from scipy.ndimage import label, binary_erosion, binary_dilation

from shelterbelts.utils import visualise_categories, visualise_categories_sidebyside, tif_categorical, get_example_data

import matplotlib.pyplot as plt

"""Tree categorization module.

This module provides functionality to categorize woody vegetation into different
tree types based on landscape ecology principles (Fragstats approach). Trees are
classified into:

- **Scattered Trees**: Individual trees or small clusters below the minimum patch size
- **Patch Core**: Interior areas of tree patches, away from edges
- **Patch Edge**: Peripheral areas of tree patches
- **Other Trees**: Corridors connecting patches or larger clusters
"""

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
    14:'Other Trees',
    0:'Not Trees'
}
inverted_labels = {v: k for k, v in tree_categories_labels.items()}


def tree_clusters(woody_veg, max_gap_size=2):
    """Assign cluster labels to trees within a given distance of each other."""

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
    """Identify tree clusters smaller than minimum patch size."""
    counts = np.bincount(trees_labelled.ravel())
    small_clusters = np.flatnonzero(counts < min_patch_size)
    scattered_area = np.isin(trees_labelled, small_clusters)
    return scattered_area


# +
def core_trees(woody_veg, edge_size=3, min_core_size=200, strict_core_area=False):
    """Identify core areas of tree patches using morphological operations."""    
    y, x = np.ogrid[-edge_size:edge_size+1, -edge_size:edge_size+1]
    core_kernel = (x**2 + y**2 <= edge_size**2)

    if strict_core_area:
        # Erode > Dilate > Erode. This method is strict because it requires the core areas to be fully connected.
        all_core_area = binary_erosion(woody_veg, structure=core_kernel, border_value=True)  # Just eroding is the most strict, and means you get large 'edge' circles from single pixels within a core area
        y, x = np.ogrid[-(edge_size+1):(edge_size+1)+1, -(edge_size+1):(edge_size+1)+1]
        core_kernel_plus_one = (x**2 + y**2 <= (edge_size+1)**2)
        y, x = np.ogrid[-(edge_size*2):(edge_size*2)+1, -(edge_size*2):(edge_size*2)+1]
        double_core_kernel = (x**2 + y**2 <= (edge_size*2)**2)
        cleaned_core_area = binary_dilation(all_core_area, structure=core_kernel_plus_one) 
        cleaned_core_area = binary_erosion(cleaned_core_area, structure=double_core_kernel, border_value=True)
        all_core_area = all_core_area | cleaned_core_area 
        all_core_area = all_core_area & woody_veg

    else:
        # Dilate > Erode > Erode. This is the least strict method, and allows core areas even when the trees are a bit patchy.
        core_area = binary_dilation(woody_veg, structure=core_kernel) 
        core_area = binary_erosion(core_area, structure=core_kernel, border_value=True)
        core_area = binary_erosion(core_area, structure=core_kernel, border_value=True) 
        all_core_area = core_area & woody_veg

    # Exclude core area's smaller than the scattered trees threshold
    labeled_cores, num_features = label(all_core_area)
    
    large_cores = np.flatnonzero(np.bincount(labeled_cores.ravel()) >= min_core_size)
    if (~woody_veg).sum() > 0:
        # large_cores = large_cores[1:]  # This doesn't work if there aren't any 0 categories larger than the min_patch_size because they got removed from the last step, so we end up losing a core area instead
        large_cores = large_cores[large_cores != 0]  # Drop the 0 category for non-trees
    core_area = np.isin(labeled_cores, large_cores) & woody_veg

    return core_area, core_kernel

# core_area, core_kernel = core_trees(woody_veg, edge_size, min_patch_size)
# plt.imshow(core_area)


# -

def tree_categories(input_data, outdir='.', stub=None, min_patch_size=20, min_core_size=1000, edge_size=3, max_gap_size=2, strict_core_area=True, save_tif=True, plot=True):
    """
    Classifies a boolean woody vegetation map into four categories based on the
    Fragstats landscape ecology approach:
    
    - **Scattered Trees** (11): Individual trees or clusters below minimum patch size
    - **Patch Core** (12): Interior areas of tree patches with buffer from edges
    - **Patch Edge** (13): Perimeter areas of tree patches within edge distance
    - **Other Trees** (14): Corridor pixels connecting patches
    
    Parameters
    ----------
    input_data : str or xarray.Dataset
        Either a file path to a binary GeoTIFF containing tree/no-tree information,
        or an xarray Dataset with a 'woody_veg' band (boolean or integer).
    outdir : str, optional
        Output directory for saving results. Default is current directory.
    stub : str, optional
        Prefix for output filenames. If input_data is a string and stub is not
        provided, it is derived from the input filename. Required when
        input_data is a Dataset.
    min_patch_size : int, optional
        Minimum area (pixels) to classify as a patch rather than scattered trees.
        Default is 20.
    min_core_size : int, optional
        Minimum area (pixels) to classify as a core area. Default is 1000.
    edge_size : int, optional
        Distance (pixels) defining the edge region around patch cores.
        Default is 3.
    max_gap_size : int, optional
        Maximum gap (pixels) to bridge when connecting tree clusters.
        Default is 2.
    strict_core_area : bool, optional
        If True, enforce that core areas are strictly connected.
        If False, use relaxed connectivity rules. Default is True.
    save_tif : bool, optional
        Whether to save the categorized result as a GeoTIFF file.
        Default is True.
    plot : bool, optional
        Whether to generate a PNG visualization of the results.
        Default is True.
    
    Returns
    -------
    xarray.Dataset
        Dataset containing:
        
        - **woody_veg**: Original binary tree/no-tree classification
        - **tree_categories**: Categorized tree types (values 0, 11, 12, 13, 14)
    
    Notes
    -----
    The categorization uses morphological operations with circular kernels defined
    by edge_size and max_gap_size. The process:
    
    1. Connects nearby tree clusters using max_gap_size
    2. Identifies core areas using edge_size
    3. Classifies remaining trees as edge or corridor
    4. Marks very small clusters as scattered trees
    
    Output Files
    ~~~~~~~~~~~~
    When save_tif=True, generates a GeoTIFF file with embedded color map:
    ``{stub}_categorised.tif``
    
    When plot=True, generates a PNG visualization with legend:
    ``{stub}_categorised.png``
    
    References
    ----------
    McGarigal, K., & Marks, B. J. (1995). FRAGSTATS: Spatial Pattern Analysis
    Program for Quantifying Landscape Structure. General Technical Report.
    
    Examples
    --------
    Creating tree categories from test data:
    
    >>> from shelterbelts.utils import create_test_woody_veg_dataset
    >>> ds_input = create_test_woody_veg_dataset()
    >>> ds_cat = tree_categories(ds_input, stub='test', outdir='/tmp', plot=False, save_tif=False)
    >>> set(ds_cat.data_vars) == {'woody_veg', 'tree_categories'}
    True
    
    Here's how different parameters affect the categorization:
    
    .. plot::
        
        from shelterbelts.indices.tree_categories import tree_categories, tree_categories_cmap, tree_categories_labels
        from shelterbelts.utils import visualise_categories_sidebyside, get_example_data
        import matplotlib.pyplot as plt
        
        test_filename = get_example_data('g2_26729_binary_tree_cover_10m.tiff')
        
        # edge_size: 1 vs 5
        ds1 = tree_categories(test_filename, edge_size=1)
        ds2 = tree_categories(test_filename, edge_size=5)
        visualise_categories_sidebyside(
            ds1['tree_categories'], ds2['tree_categories'],
            colormap=tree_categories_cmap, labels=tree_categories_labels,
            title1="edge_size=1", title2="edge_size=5"
        )
        
        # min_patch_size: 10 vs 30
        ds1 = tree_categories(test_filename, min_patch_size=10)
        ds2 = tree_categories(test_filename, min_patch_size=30)
        visualise_categories_sidebyside(
            ds1['tree_categories'], ds2['tree_categories'],
            colormap=tree_categories_cmap, labels=tree_categories_labels,
            title1="min_patch_size=10", title2="min_patch_size=30"
        )
        
        # max_gap_size: 0 vs 2
        ds1 = tree_categories(test_filename, max_gap_size=0)
        ds2 = tree_categories(test_filename, max_gap_size=2)
        visualise_categories_sidebyside(
            ds1['tree_categories'], ds2['tree_categories'],
            colormap=tree_categories_cmap, labels=tree_categories_labels,
            title1="max_gap_size=0", title2="max_gap_size=2"
        )
        
        # strict_core_area: False vs True
        ds1 = tree_categories(test_filename, strict_core_area=False)
        ds2 = tree_categories(test_filename, strict_core_area=True)
        visualise_categories_sidebyside(
            ds1['tree_categories'], ds2['tree_categories'],
            colormap=tree_categories_cmap, labels=tree_categories_labels,
            title1="strict_core_area=False", title2="strict_core_area=True"
        )
    
    """
    if isinstance(input_data, str):
        da = rxr.open_rasterio(input_data).isel(band=0).drop_vars('band')
        ds = da.to_dataset(name='woody_veg')
        filename = input_data
    else:
        ds = input_data
        filename = None

    woody_veg = ds['woody_veg'].values.astype(bool)

    trees_labelled = tree_clusters(woody_veg, max_gap_size)
    scattered_area = scattered_trees(trees_labelled, min_patch_size)
    core_area, core_kernel = core_trees(woody_veg, edge_size, min_core_size, strict_core_area)
    edge_kernel = np.ones(core_kernel.shape, dtype=bool)
    edge_area = binary_dilation(core_area, structure=edge_kernel) & ~core_area & woody_veg
    corridor_area = woody_veg & ~(core_area | edge_area | scattered_area)

    tree_categories_array = np.zeros_like(woody_veg, dtype=np.uint8)
    tree_categories_array[scattered_area] = inverted_labels['Scattered Trees']
    tree_categories_array[core_area]      = inverted_labels['Patch Core']
    tree_categories_array[edge_area]      = inverted_labels['Patch Edge']
    tree_categories_array[corridor_area]  = inverted_labels['Other Trees']
    ds['tree_categories'] = (('y', 'x'), tree_categories_array)
    # ds = ds.rename({'x':'longitude', 'y': 'latitude'})

    if not stub:
        if filename:
            # Use the same prefix as the original woody_veg filename
            stub = filename.split('/')[-1].split('.')[0]
        else:
            raise ValueError("stub must be provided when input_data is a Dataset")

    if save_tif:
        filename_categorical = os.path.join(outdir,f"{stub}_categorised.tif")
        tif_categorical(ds['tree_categories'], filename_categorical, tree_categories_cmap)

    if plot:
        filename_categorical_png = os.path.join(outdir, f"{stub}_categorised.png")
        # filename_categorical_png = None
        visualise_categories(ds['tree_categories'], filename_categorical_png, tree_categories_cmap, tree_categories_labels, "Tree Categories")
                
    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description='Categorize woody vegetation into tree types using landscape ecology methods.'
    )
    
    parser.add_argument('--filename', help='A binary tif file containing tree/no tree information')
    parser.add_argument('--outdir', default='.', help='The output directory to save the results')
    parser.add_argument('--stub', default=None, help='Prefix for output files.')
    parser.add_argument('--min_patch_size', default=20, type=int, help='The minimum area to be classified as a patch/corrider rather than just scattered trees.')
    parser.add_argument('--edge_size', default=3, type=int, help='The buffer distance at the edge of a patch, with pixels inside this being the core area')
    parser.add_argument('--max_gap_size', default=2, type=int, help='The allowable gap between two tree clusters before considering them as separate patches.')
    parser.add_argument('--no-strict-core-area', dest='strict_core_area', action="store_false", default=True, help='Disable strict core area enforcement (default: enabled)')
    parser.add_argument('--no-plot', dest='plot', action="store_false", default=True, help='Disable PNG visualization (default: enabled)')
 
    return parser


if __name__ == '__main__':

    parser = parse_arguments()
    args = parser.parse_args()
    
    filename = args.filename
    outdir = args.outdir
    stub = args.stub
    min_patch_size = args.min_patch_size
    edge_size = args.edge_size
    max_gap_size = args.max_gap_size
    plot = args.plot
    
    tree_categories(filename, outdir, stub, min_patch_size, edge_size, max_gap_size, args.strict_core_area, plot=plot)

# +
# # %%time
# cover_threshold=50
# min_patch_size=20
# min_core_size=200
# edge_size=10
# max_gap_size=1
# strict_core_area=False

# folder='/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_30_lon_158'
# tmpdir = '/scratch/xe2/cb8590/tmp'
# outdir='/scratch/xe2/cb8590/tmp'
# stub='Test'

# cover_threshold=50
# # percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_28_lon_144/29_33-144_02_y2024_predicted.tif'  # Failing because all trees
# # percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2024/subfolders/lat_34_lon_140/34_13-141_90_y2024_predicted.tif' # Should be fine
# percent_tif = '/scratch/xe2/cb8590/barra_trees_s4_2018_actnsw_4326/subfolders/lat_34_lon_148/34_37-148_42_y2018_predicted.tif'  # West Milgadara
# da_percent = rxr.open_rasterio(percent_tif).isel(band=0).drop_vars('band')
# da_trees = da_percent > cover_threshold
# da_trees = da_trees.astype('uint8')
# ds_woody_veg = da_trees.to_dataset(name='woody_veg')
# -

# ds_tree_categories = tree_categories(None, outdir, stub, min_core_size=1000, edge_size=10, strict_core_area=True, min_patch_size=min_patch_size, max_gap_size=max_gap_size, save_tif=True, plot=True, ds=ds_woody_veg)



