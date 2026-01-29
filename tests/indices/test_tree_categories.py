import os
import pytest

from shelterbelts.indices.tree_categories import tree_categories


# Configuration
stub = 'g2_26729'
test_filename = f'data/{stub}_binary_tree_cover_10m.tiff'


def test_tree_categories_basic():
    """Basic test for tree_categories function"""
    ds = tree_categories(test_filename, outdir='outdir', stub=stub)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert os.path.exists(f"outdir/{stub}_categorised.tif")
    assert os.path.exists(f"outdir/{stub}_categorised.png")


def test_tree_categories_patch_size():
    """Test tree_categories with different patch sizes"""
    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_patch50', min_patch_size=50)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}


def test_tree_categories_edge_size():
    """Test tree_categories with different edge sizes"""
    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_edge10', edge_size=10)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}


def test_tree_categories_gap_size():
    """Test tree_categories with different max_gap_size"""
    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_gap0', max_gap_size=0)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}


def test_tree_categories_no_save():
    """Test tree_categories without saving tif"""
    if os.path.exists(f"outdir/{stub}_categorised.tif"):
        os.remove(f"outdir/{stub}_categorised.tif")
    
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, save_tif=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_categorised.tif")


def test_tree_categories_no_plot():
    """Test tree_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_categorised.png"):
        os.remove(f"outdir/{stub}_categorised.png")
    
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_categorised.png")
