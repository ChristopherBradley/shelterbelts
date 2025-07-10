import os
import time

from shelterbelts.indices.tree_categories import tree_categories

if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')

# Should add this to git so the tests can run after doing a git clone
test_filename = 'data/g2_26729_binary_tree_cover_10m.tiff'

def test_basic():
    """Basic tests for each of the files"""
    ds = tree_categories(test_filename, outdir='outdir', stub='TEST', min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert os.path.exists("outdir/TEST_categorised.tif")
    assert os.path.exists("outdir/TEST_categorised.png")

def test_tree_categories():
    """More comprehensive tree category tests: 2x patch sizes, 2x edge sizes, 2x max_gap_sizes, without saving tif, without plot"""
    ds = tree_categories(test_filename, outdir='outdir', stub='TEST_patch50', min_patch_size=50, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    ds = tree_categories(test_filename, outdir='outdir', stub='TEST_edge10', min_patch_size=10, edge_size=10, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    ds = tree_categories(test_filename, outdir='outdir', stub='TEST_gap0', min_patch_size=10, edge_size=3, max_gap_size=0, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    if os.path.exists("outdir/TEST_categorised.tif"):
        os.remove("outdir/TEST_categorised.tif")
    ds = tree_categories(test_filename, outdir='outdir', stub='TEST', min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=False, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists("outdir/TEST_categorised.tif")

    if os.path.exists("outdir/TEST_categorised.png"):
        os.remove("outdir/TEST_categorised.png")
    ds = tree_categories(test_filename, outdir='outdir', stub='TEST', min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists("outdir/TEST_categorised.png")

if __name__ == '__main__':
    test_basic()
    test_tree_categories()