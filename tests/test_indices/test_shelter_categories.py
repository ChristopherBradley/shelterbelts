import os
import pytest
import numpy as np
import xarray as xr

from shelterbelts.indices.shelter_categories import compute_distance_to_tree_TH, shelter_categories
from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.utils.filepaths import create_test_woody_veg_dataset


stub = 'g2_26729'
test_filename = f"data/{stub}_tree_categories.tif"
wind_file = f"data/{stub}_barra_daily.nc"
height_file = f"data/{stub}_canopy_height.tif"


def _assert_shelter_output(ds):
    """Helper to assert expected output structure"""
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)


def test_shelter_categories_basic():
    """Basic test for shelter_categories function"""
    ds = shelter_categories(test_filename, outdir='outdir', stub=stub)
    _assert_shelter_output(ds)
    assert os.path.exists(f"outdir/{stub}_shelter_categories.tif")
    assert os.path.exists(f"outdir/{stub}_shelter_categories.png")

# TODO: Add a small chm.tif to the data folder (the current on in outdir is larger than I'd like to add to git)
# @pytest.mark.parametrize("minimum_height,suffix", [(1, "minCH1"), (5, "minCH5"), (10, "minCH10")])
# def test_shelter_categories_minimum_height(minimum_height, suffix):
#     """Test shelter_categories with different minimum heights"""
#     ds = shelter_categories(
#         test_filename,
#         wind_data=wind_file,
#         height_tif=height_file,
#         outdir='outdir',
#         stub=f'{stub}_{suffix}',
#         minimum_height=minimum_height,
#         plot=False
#     )
#     _assert_shelter_output(ds)


@pytest.mark.parametrize("wind_method,suffix", [
    ("MAX", "methodMAX"),
    ("ANY", "methodANY"),
    ("MOST_COMMON", "methodCOMMON"),
    ("HAPPENED", "methodHAPPENED"),
])
def test_shelter_categories_wind_methods(wind_method, suffix):
    """Test shelter_categories with different wind methods"""
    ds = shelter_categories(
        test_filename,
        wind_data=wind_file,
        outdir='outdir',
        stub=f'{stub}_{suffix}',
        wind_method=wind_method,
        plot=False
    )
    _assert_shelter_output(ds)


@pytest.mark.parametrize("wind_threshold,suffix", [(10, "w10"), (20, "w20"), (25, "w25"), (30, "w30")])
def test_shelter_categories_wind_threshold(wind_threshold, suffix):
    """Test shelter_categories with different wind speed thresholds"""
    ds = shelter_categories(
        test_filename,
        wind_data=wind_file,
        outdir='outdir',
        stub=f'{stub}_{suffix}',
        wind_method='HAPPENED',
        wind_threshold=wind_threshold,
        plot=False
    )
    _assert_shelter_output(ds)


@pytest.mark.parametrize("distance_threshold,suffix", [(30, "d30"), (15, "d15"), (10, "d10")])
def test_shelter_categories_distance_threshold(distance_threshold, suffix):
    """Test shelter_categories with different distance thresholds"""
    ds = shelter_categories(
        test_filename,
        wind_data=wind_file,
        outdir='outdir',
        stub=f'{stub}_{suffix}',
        distance_threshold=distance_threshold,
        plot=False
    )
    _assert_shelter_output(ds)


# TODO: Add a small chm.tif to the data folder (the current on in outdir is larger than I'd like to add to git)
# @pytest.mark.parametrize("distance_threshold,suffix", [(30, "dCH30"), (15, "dCH15"), (10, "dCH10")])
# def test_shelter_categories_distance_with_height(distance_threshold, suffix):
#     """Test shelter_categories with distance thresholds and height tif"""
#     ds = shelter_categories(
#         test_filename,
#         wind_data=wind_file,
#         height_tif=height_file,
#         outdir='outdir',
#         stub=f'{stub}_{suffix}',
#         distance_threshold=distance_threshold,
#         minimum_height=1,
#         plot=False
#     )
#     _assert_shelter_output(ds)


@pytest.mark.parametrize("density_threshold,suffix", [(5, "density5"), (20, "density20")])
def test_shelter_categories_density_threshold(density_threshold, suffix):
    """Test shelter_categories with different density thresholds"""
    ds = shelter_categories(
        test_filename,
        outdir='outdir',
        stub=f'{stub}_{suffix}',
        density_threshold=density_threshold,
        plot=False
    )
    _assert_shelter_output(ds)


def test_shelter_categories_no_save():
    """Test shelter_categories without saving tif"""
    if os.path.exists(f"outdir/{stub}_shelter_categories.tif"):
        os.remove(f"outdir/{stub}_shelter_categories.tif")
    
    ds = shelter_categories(test_filename, outdir='outdir', stub=stub, savetif=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.tif")


def test_shelter_categories_no_plot():
    """Test shelter_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_shelter_categories.png"):
        os.remove(f"outdir/{stub}_shelter_categories.png")
    
    ds = shelter_categories(test_filename, outdir='outdir', stub=stub, plot=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.png")


def test_shelter_categories_from_dataset():
    """Test shelter_categories using an in-memory dataset."""
    ds_input = create_test_woody_veg_dataset()
    ds_cat = tree_categories(ds_input, stub='test', outdir='outdir', plot=False, save_tif=False)

    ds = shelter_categories(ds_cat, outdir='outdir', stub=f"{stub}_ds", plot=False, savetif=False)
    assert 'shelter_categories' in set(ds.data_vars)

# Convert 1D lists to 2D xarray DataArrays with x and y dimensions
def to_dataarray(heights):
    arr = np.array([heights], dtype=float)  # Create 2D array (1, n) with float dtype
    return xr.DataArray(arr, dims=['y', 'x'])

def test_tree_height_method():
    """Test the shelter categories using tree heights.
    Each tree of height h shelters h pixels downwind (wind_dir='E' by default).
    Returns the distance to the nearest upwind tree that can shelter each pixel.
    NaN for tree pixels and unsheltered pixels.
    """
    N = np.nan
    heights_small_trees =           to_dataarray([1,0,0,1,0,0,0])
    heights_single_tree =           to_dataarray([0,3,0,0,0,0,0])
    heights_two_trees =             to_dataarray([0,4,1,0,0,0,0])
    heights_two_trees_separated =   to_dataarray([0,4,0,1,0,0,0])

    expected_small_trees =          to_dataarray([N,1,N,N,1,N,N])
    expected_single_tree =          to_dataarray([N,N,1,2,3,N,N])
    expected_two_trees =            to_dataarray([N,N,N,1,3,4,N])
    expected_two_trees_separated =  to_dataarray([N,N,1,N,1,4,N])

    xr.testing.assert_equal(compute_distance_to_tree_TH(heights_small_trees), expected_small_trees)
    xr.testing.assert_equal(compute_distance_to_tree_TH(heights_single_tree), expected_single_tree)
    xr.testing.assert_equal(compute_distance_to_tree_TH(heights_two_trees), expected_two_trees)
    xr.testing.assert_equal(compute_distance_to_tree_TH(heights_two_trees_separated), expected_two_trees_separated)

def test_pytest():
    print("Hello world!")