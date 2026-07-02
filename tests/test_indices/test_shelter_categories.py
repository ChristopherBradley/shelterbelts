import os
import pytest
import numpy as np
import xarray as xr
import rioxarray as rxr

from shelterbelts.indices.shelter_categories import (
    compute_distance_to_tree_TH,
    shelter_categories,
    shelter_categories_labels,
    direction_map,
)


stub = 'g2_26729'
linear_file = f"data/{stub}_linear_categories.tif"
wind_file = f"data/{stub}_barra_daily.nc"
height_file = f"data/{stub}_chm_res10_filled.tif"


def _assert_shelter_output(ds):
    """Helper to assert expected output structure"""
    assert 'shelter_categories' in set(ds.data_vars)
    assert set(ds['shelter_categories'].coords) >= {'x', 'y', 'spatial_ref'}


def test_shelter_categories_basic():
    """Basic test for shelter_categories function (density method, no wind)."""
    ds = shelter_categories(linear_file, outdir='outdir', stub=stub)
    _assert_shelter_output(ds)
    assert os.path.exists(f"outdir/{stub}_shelter_categories.tif")
    assert os.path.exists(f"outdir/{stub}_shelter_categories.png")


@pytest.mark.parametrize("wind_method", ["MAX", "ANY", "MOST_COMMON", "HAPPENED", "WINDWARD"])
def test_shelter_categories_wind_methods(wind_method):
    """Every wind method produces valid, labelled categories."""
    ds = shelter_categories(
        linear_file, wind_data=wind_file, outdir='outdir',
        stub=f'{stub}_{wind_method}', wind_method=wind_method, plot=False,
    )
    _assert_shelter_output(ds)
    values = set(int(v) for v in np.unique(ds['shelter_categories'].values))
    assert values <= set(shelter_categories_labels), f"Undefined categories: {values - set(shelter_categories_labels)}"


@pytest.mark.parametrize("wind_threshold", [10, 20, 25, 30])
def test_shelter_categories_wind_threshold(wind_threshold):
    ds = shelter_categories(
        linear_file, wind_data=wind_file, outdir='outdir', stub=f'{stub}_w{wind_threshold}',
        wind_method='HAPPENED', wind_threshold=wind_threshold, plot=False,
    )
    _assert_shelter_output(ds)


@pytest.mark.parametrize("distance_threshold", [30, 15, 10])
def test_shelter_categories_distance_threshold(distance_threshold):
    ds = shelter_categories(
        linear_file, wind_data=wind_file, outdir='outdir', stub=f'{stub}_d{distance_threshold}',
        distance_threshold=distance_threshold, plot=False,
    )
    _assert_shelter_output(ds)


@pytest.mark.parametrize("density_threshold", [5, 20])
def test_shelter_categories_density_threshold(density_threshold):
    """Density method (no wind): sheltered farmland is the generic 32/42."""
    ds = shelter_categories(
        linear_file, outdir='outdir', stub=f'{stub}_density{density_threshold}',
        density_threshold=density_threshold, plot=False,
    )
    _assert_shelter_output(ds)
    values = set(int(v) for v in np.unique(ds['shelter_categories'].values))
    # Density can't attribute a tree type, so no 33-39/43-49 codes appear
    assert values & set(range(33, 40)) == set()
    assert values & set(range(43, 50)) == set()


def test_shelter_categories_height_tif():
    """Providing a canopy height tif still produces valid output."""
    ds = shelter_categories(
        linear_file, wind_data=wind_file, height_tif=height_file, wind_method='MOST_COMMON',
        outdir='outdir', stub=f'{stub}_height', plot=False, savetif=False,
    )
    _assert_shelter_output(ds)


def test_shelter_categories_conserves_farmland():
    """Re-encoding must not move pixels between grassland (30-39) and cropland (40-49)."""
    da_in = rxr.open_rasterio(linear_file).isel(band=0)
    n_grass = int(da_in.isin([30, 31, 32]).sum())
    n_crop = int(da_in.isin([40, 41, 42]).sum())

    ds = shelter_categories(linear_file, wind_data=wind_file, wind_method='WINDWARD',
                            outdir='outdir', stub=f'{stub}_conserve', savetif=False, plot=False)
    da_out = ds['shelter_categories']
    assert int(((da_out >= 30) & (da_out < 40)).sum()) == n_grass
    assert int(((da_out >= 40) & (da_out < 50)).sum()) == n_crop


def test_shelter_categories_preserves_trees():
    """Tree pixels (10-19) must be unchanged."""
    da_in = rxr.open_rasterio(linear_file).isel(band=0)
    tree_mask = ((da_in >= 10) & (da_in < 20)).values

    ds = shelter_categories(linear_file, wind_data=wind_file, wind_method='ANY',
                            outdir='outdir', stub=f'{stub}_trees', savetif=False, plot=False)
    da_out = ds['shelter_categories']
    assert bool(((da_out.values >= 10) & (da_out.values < 20) == tree_mask).all())
    assert np.array_equal(da_out.values[tree_mask], da_in.values[tree_mask])


def test_shelter_categories_no_save():
    if os.path.exists(f"outdir/{stub}_shelter_categories.tif"):
        os.remove(f"outdir/{stub}_shelter_categories.tif")
    shelter_categories(linear_file, outdir='outdir', stub=stub, savetif=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.tif")


def test_shelter_categories_no_plot():
    if os.path.exists(f"outdir/{stub}_shelter_categories.png"):
        os.remove(f"outdir/{stub}_shelter_categories.png")
    shelter_categories(linear_file, outdir='outdir', stub=stub, plot=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.png")


def test_shelter_categories_multi_layer():
    """MULTI_LAYER returns an 8-band shelter_distances output in clockwise order."""
    ds = shelter_categories(
        linear_file, wind_method='MULTI_LAYER', outdir='outdir', stub=f'{stub}_multilayer',
        savetif=True, plot=False,
    )
    assert 'shelter_distances' in ds.data_vars
    assert 'shelter_categories' in ds.data_vars
    assert ds['shelter_distances'].sizes['direction'] == 8

    expected_order = list(direction_map.keys())
    assert ds['shelter_distances'].direction.values.tolist() == expected_order

    da_saved = rxr.open_rasterio(f'outdir/{stub}_multilayer_shelter_categories.tif')
    assert da_saved.shape[0] == 8


def test_shelter_categories_from_dataset():
    """Accepts a Dataset carrying the linear_categories band and adds the new band alongside it."""
    da = rxr.open_rasterio(linear_file).squeeze('band').drop_vars('band')
    ds_linear = da.to_dataset(name='linear_categories')
    ds = shelter_categories(ds_linear, wind_data=wind_file, wind_method='ANY',
                            outdir='outdir', stub=f"{stub}_ds", plot=False, savetif=False)
    assert {'linear_categories', 'shelter_categories'} <= set(ds.data_vars)


# Convert 1D lists to 2D xarray DataArrays with x and y dimensions
def to_dataarray(heights):
    arr = np.array([heights], dtype=float)  # Create 2D array (1, n) with float dtype
    return xr.DataArray(arr, dims=['y', 'x'])


def test_tree_height_method():
    """Each tree of height h shelters h pixels downwind (wind_dir='E' by default).
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
