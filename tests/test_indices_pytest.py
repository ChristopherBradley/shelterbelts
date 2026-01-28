import os
import pytest

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics


@pytest.fixture(scope="session", autouse=True)
def setup_dirs():
    """Create necessary directories for tests"""
    if not os.path.exists('tmpdir'):
        os.mkdir('tmpdir')
    if not os.path.exists('outdir'):
        os.mkdir('outdir')


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


def test_cover_categories_basic():
    """Basic test for cover_categories function"""
    ds = cover_categories(
        f"outdir/{stub}_shelter_categories.tif",
        f"outdir/{stub}_worldcover.tif",
        outdir='outdir',
        stub=stub
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'cover_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_cover_categories.tif")
    assert os.path.exists(f"outdir/{stub}_cover_categories.png")


def test_cover_categories_no_save():
    """Test cover_categories without saving tif"""
    if os.path.exists(f"outdir/{stub}_cover_categories.tif"):
        os.remove(f"outdir/{stub}_cover_categories.tif")
    
    ds = cover_categories(
        f"outdir/{stub}_shelter_categories.tif",
        f"outdir/{stub}_worldcover.tif",
        outdir='outdir',
        stub=stub,
        savetif=False
    )
    assert not os.path.exists(f"outdir/{stub}_cover_categories.tif")


def test_cover_categories_no_plot():
    """Test cover_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_cover_categories.png"):
        os.remove(f"outdir/{stub}_cover_categories.png")
    
    ds = cover_categories(
        f"outdir/{stub}_shelter_categories.tif",
        f"outdir/{stub}_worldcover.tif",
        outdir='outdir',
        stub=stub,
        plot=False
    )
    assert not os.path.exists(f"outdir/{stub}_cover_categories.png")


def test_buffer_categories_basic():
    """Basic test for buffer_categories function"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_buffer_categories.tif")
    assert os.path.exists(f"outdir/{stub}_buffer_categories.png")


def test_buffer_categories_buffer_width():
    """Test buffer_categories with different buffer widths"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_ridges():
    """Test buffer_categories with ridges tif"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        ridges_tif=f'outdir/{stub}_ridges.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_roads():
    """Test buffer_categories with roads tif"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        roads_tif=f'outdir/{stub}_roads.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_ridges_and_roads():
    """Test buffer_categories with both ridges and roads tif"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        ridges_tif=f'outdir/{stub}_ridges.tif',
        roads_tif=f'outdir/{stub}_roads.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_no_save():
    """Test buffer_categories without saving tif"""
    if os.path.exists(f"outdir/{stub}_buffer_categories.tif"):
        os.remove(f"outdir/{stub}_buffer_categories.tif")
    
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub,
        savetif=False
    )
    assert not os.path.exists(f"outdir/{stub}_buffer_categories.tif")


def test_buffer_categories_no_plot():
    """Test buffer_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_buffer_categories.png"):
        os.remove(f"outdir/{stub}_buffer_categories.png")
    
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub,
        plot=False
    )
    assert not os.path.exists(f"outdir/{stub}_buffer_categories.png")


def test_patch_metrics_basic():
    """Basic test for patch_metrics function"""
    ds, df = patch_metrics(
        f"outdir/{stub}_buffer_categories.tif",
        outdir="outdir",
        stub=stub
    )
    assert os.path.exists(f"outdir/{stub}_linear_categories.tif")
    assert os.path.exists(f"outdir/{stub}_linear_categories.png")
    assert os.path.exists(f"outdir/{stub}_labelled_categories.tif")
    assert os.path.exists(f"outdir/{stub}_labelled_categories.png")
    assert os.path.exists(f"outdir/{stub}_patch_metrics.csv")


def test_class_metrics_basic():
    """Basic test for class_metrics function"""
    dfs = class_metrics(
        f"outdir/{stub}_linear_categories.tif",
        outdir="outdir",
        stub=stub,
        save_excel=True
    )
    assert os.path.exists(f"outdir/{stub}_class_metrics.xlsx")
