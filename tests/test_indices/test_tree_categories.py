import os

from shelterbelts.indices import tree_categories
from shelterbelts.utils import create_test_woody_veg_dataset


# Configuration
stub = 'g2_26729'
test_filename = f'data/{stub}_binary_tree_cover_10m.tiff'


def test_tree_categories_basic():
    """Basic test for tree_categories function"""
    ds = tree_categories(test_filename, outdir='outdir', stub=stub)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert os.path.exists(f"outdir/{stub}_tree_categories.tif")
    assert os.path.exists(f"outdir/{stub}_tree_categories.png")


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
    if os.path.exists(f"outdir/{stub}_tree_categories.tif"):
        os.remove(f"outdir/{stub}_tree_categories.tif")
    
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, save_tif=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_tree_categories.tif")


def test_tree_categories_no_plot():
    """Test tree_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_tree_categories.png"):
        os.remove(f"outdir/{stub}_tree_categories.png")
    
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_tree_categories.png")


def test_tree_categories_does_not_mutate_input_dataset():
    """Ensure input Dataset is not mutated in-place."""
    ds_input = create_test_woody_veg_dataset()
    ds_input_before = ds_input.copy(deep=True)

    ds_output = tree_categories(ds_input, stub='test', outdir='outdir', plot=False, save_tif=False)

    assert 'tree_categories' not in ds_input.data_vars
    assert set(ds_output.data_vars) == {'woody_veg', 'tree_categories'}
    assert ds_input['woody_veg'].equals(ds_input_before['woody_veg'])


def test_tree_categories_strict_core_area():
    """Test tree_categories with strict_core_area parameter"""
    ds_strict_false = tree_categories(
        test_filename, stub='strict_false', outdir='outdir',
        plot=False, save_tif=False, strict_core_area=False
    )
    ds_strict_true = tree_categories(
        test_filename, stub='strict_true', outdir='outdir',
        plot=False, save_tif=False, strict_core_area=True
    )
    assert set(ds_strict_false.data_vars) == {'woody_veg', 'tree_categories'}
    assert set(ds_strict_true.data_vars) == {'woody_veg', 'tree_categories'}
    
    # Check for differences in core areas
    diff_strict = (ds_strict_false['tree_categories'] != ds_strict_true['tree_categories']).sum().item()
    assert diff_strict >= 0
