import os
import pytest

from shelterbelts.indices.shelter_categories import shelter_categories


stub = 'g2_26729'


def test_shelter_categories_basic():
    """Basic test for shelter_categories function"""
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", outdir='outdir', stub=stub)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_shelter_categories.tif")
    assert os.path.exists(f"outdir/{stub}_shelter_categories.png")


def test_shelter_categories_minimum_height():
    """Test shelter_categories with different minimum heights"""
    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        height_tif=f"outdir/{stub}_canopy_height.tif",
        outdir='outdir',
        stub=f'{stub}_minCH1',
        minimum_height=1,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        height_tif=f"outdir/{stub}_canopy_height.tif",
        outdir='outdir',
        stub=f'{stub}_minCH5',
        minimum_height=5,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        height_tif=f"outdir/{stub}_canopy_height.tif",
        outdir='outdir',
        stub=f'{stub}_minCH10',
        minimum_height=10,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)


def test_shelter_categories_wind_methods():
    """Test shelter_categories with different wind methods"""
    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_methodMAX',
        wind_method='MAX',
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_methodANY',
        wind_method='ANY',
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_methodCOMMON',
        wind_method='MOST_COMMON',
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_methodHAPPENED',
        wind_method='HAPPENED',
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)


def test_shelter_categories_wind_threshold():
    """Test shelter_categories with different wind speed thresholds"""
    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_w10',
        wind_method='HAPPENED',
        wind_threshold=10,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_w20',
        wind_method='HAPPENED',
        wind_threshold=20,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_w25',
        wind_method='HAPPENED',
        wind_threshold=25,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_w30',
        wind_method='HAPPENED',
        wind_threshold=30,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)


def test_shelter_categories_distance_threshold():
    """Test shelter_categories with different distance thresholds"""
    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_d30',
        distance_threshold=30,
        minimum_height=1,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_d15',
        distance_threshold=15,
        minimum_height=1,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        outdir='outdir',
        stub=f'{stub}_d10',
        distance_threshold=10,
        minimum_height=1,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)


def test_shelter_categories_distance_with_height():
    """Test shelter_categories with distance thresholds and height tif"""
    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        height_tif=f"outdir/{stub}_canopy_height.tif",
        outdir='outdir',
        stub=f'{stub}_dCH30',
        distance_threshold=30,
        minimum_height=1,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        height_tif=f"outdir/{stub}_canopy_height.tif",
        outdir='outdir',
        stub=f'{stub}_dCH15',
        distance_threshold=15,
        minimum_height=1,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        wind_nc=f"outdir/{stub}_barra_daily.nc",
        height_tif=f"outdir/{stub}_canopy_height.tif",
        outdir='outdir',
        stub=f'{stub}_dCH10',
        distance_threshold=10,
        minimum_height=1,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)


def test_shelter_categories_density_threshold():
    """Test shelter_categories with different density thresholds"""
    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        outdir='outdir',
        stub=f'{stub}_density5',
        density_threshold=5,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(
        f"outdir/{stub}_categorised.tif",
        outdir='outdir',
        stub=f'{stub}_density20',
        density_threshold=20,
        plot=False
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'shelter_categories' in set(ds.data_vars)


def test_shelter_categories_no_save():
    """Test shelter_categories without saving tif"""
    if os.path.exists(f"outdir/{stub}_shelter_categories.tif"):
        os.remove(f"outdir/{stub}_shelter_categories.tif")
    
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", outdir='outdir', stub=stub, savetif=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.tif")


def test_shelter_categories_no_plot():
    """Test shelter_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_shelter_categories.png"):
        os.remove(f"outdir/{stub}_shelter_categories.png")
    
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", outdir='outdir', stub=stub, plot=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.png")
