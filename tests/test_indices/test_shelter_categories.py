import os
import pytest

from shelterbelts.indices import shelter_categories, tree_categories
from shelterbelts.utils import create_test_woody_veg_dataset


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
        minimum_height=1,
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
