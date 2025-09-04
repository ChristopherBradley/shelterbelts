import os
import time

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.buffer_categories import buffer_categories
from shelterbelts.indices.shelter_metrics import class_metrics, patch_metrics

if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')

# Should add this to git so the tests can run after doing a git clone. And/or a .laz file that can be downloaded from ELVIS.
stub = 'g2_26729'
test_filename = f'data/{stub}_binary_tree_cover_10m.tiff'


def test_basic():
    """Basic tests for each of the files"""
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert os.path.exists(f"outdir/{stub}_categorised.tif")
    assert os.path.exists(f"outdir/{stub}_categorised.png")

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=stub, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_shelter_categories.tif")
    assert os.path.exists(f"outdir/{stub}_shelter_categories.png")

    ds = cover_categories(f"outdir/{stub}_shelter_categories.tif", f"outdir/{stub}_worldcover.tif", outdir='outdir', stub=stub, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'cover_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_cover_categories.tif")
    assert os.path.exists(f"outdir/{stub}_cover_categories.png")

    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=None, roads_tif=None, outdir="outdir", stub=stub, buffer_width=3, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'buffer_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_buffer_categories.tif")
    assert os.path.exists(f"outdir/{stub}_buffer_categories.png")


def test_tree_categories():
    """More comprehensive tree category tests: 2x patch sizes, 2x edge sizes, 2x max_gap_sizes, without saving tif, without plot"""
    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_patch50', min_patch_size=50, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_edge10', min_patch_size=10, edge_size=10, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_gap0', min_patch_size=10, edge_size=3, max_gap_size=0, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    if os.path.exists(f"outdir/{stub}_categorised.tif"):
        os.remove(f"outdir/{stub}_categorised.tif")
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=False, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_categorised.tif")

    if os.path.exists(f"outdir/{stub}_categorised.png"):
        os.remove(f"outdir/{stub}_categorised.png")
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_categorised.png")


def test_shelter_categories():
    """More comprehensive shelter category tests: with & without height tif & wind ds, 3x wind_methods, 2x wind_threshold, density_threshold, distance_threshold, minimum_height, with and without savetif and plotting"""
    # Changing the minimum height
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_minCH1', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_minCH5', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=5, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_minCH10', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the wind method
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodMAX', wind_method='MAX', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodANY', wind_method='ANY', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodCOMMON', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodHAPPENED', wind_method='HAPPENED', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the wind speed threshold
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w10', wind_method='HAPPENED', wind_threshold=10, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w20', wind_method='HAPPENED', wind_threshold=20, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w25', wind_method='HAPPENED', wind_threshold=25, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w30', wind_method='HAPPENED', wind_threshold=30, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the distance threshold
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_d30', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=30, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_d15', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=15, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_d10', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=10, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the distance threshold, with a corresponding height tif
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_dCH30', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=30, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_dCH15', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=15, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_dCH10', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=10, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the density threshold
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=f'{stub}_density5', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=5, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=f'{stub}_density20', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=20, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    if os.path.exists(f"outdir/{stub}_shelter_categories.tif"):
        os.remove(f"outdir/{stub}_shelter_categories.tif")
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=stub, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=False, plot=True)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.tif")

    if os.path.exists(f"outdir/{stub}_shelter_categories.png"):
        os.remove(f"outdir/{stub}_shelter_categories.png")
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=stub, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.png")


def test_cover_categories():
    """More comprehensive cover category tests: with and without saving an plotting"""
    if os.path.exists(f"outdir/{stub}_cover_categories.tif"):
        os.remove(f"outdir/{stub}_cover_categories.tif")
    ds = cover_categories(f"outdir/{stub}_shelter_categories.tif", f"outdir/{stub}_worldcover.tif", outdir='outdir', stub=stub, savetif=False, plot=True)
    assert not os.path.exists(f"outdir/{stub}_cover_categories.tif")

    if os.path.exists(f"outdir/{stub}_cover_categories.png"):
        os.remove(f"outdir/{stub}_cover_categories.png")
    ds = cover_categories(f"outdir/{stub}_shelter_categories.tif", f"outdir/{stub}_worldcover.tif", outdir='outdir', stub=stub, savetif=True, plot=False)
    assert not os.path.exists(f"outdir/{stub}_cover_categories.png")
    
    # Same function as in test_basic, so that I have all the outputs at the end
    ds = cover_categories(f"outdir/{stub}_shelter_categories.tif", f"outdir/{stub}_worldcover.tif", outdir='outdir', stub=stub, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'cover_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_cover_categories.tif")
    assert os.path.exists(f"outdir/{stub}_cover_categories.png")


def test_buffer_categories():
    """More comprehensive buffer category tests: 2x buffer widths, with and without ridges_tif, roads, both ridges & roads, saving, plotting"""
    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=None, roads_tif=None, outdir="outdir", stub=stub, buffer_width=5, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'buffer_categories' in set(ds.data_vars)

    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=f'outdir/{stub}_ridges.tif', roads_tif=None, outdir="outdir", stub=stub, buffer_width=5, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'buffer_categories' in set(ds.data_vars)

    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=None, roads_tif=f'outdir/{stub}_roads.tif', outdir="outdir", stub=stub, buffer_width=5, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'buffer_categories' in set(ds.data_vars)

    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=f'outdir/{stub}_ridges.tif', roads_tif=f'outdir/{stub}_roads.tif', outdir="outdir", stub=stub, buffer_width=5, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'buffer_categories' in set(ds.data_vars)

    if os.path.exists(f"outdir/{stub}_buffer_categories.tif"):
        os.remove(f"outdir/{stub}_buffer_categories.tif")
    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=None, roads_tif=None, outdir="outdir", stub=stub, buffer_width=3, savetif=False, plot=True)
    assert not os.path.exists(f"outdir/{stub}_buffer_categories.tif")

    if os.path.exists(f"outdir/{stub}_buffer_categories.png"):
        os.remove(f"outdir/{stub}_buffer_categories.png")
    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=None, roads_tif=None, outdir="outdir", stub=stub, buffer_width=3, savetif=True, plot=False)
    assert not os.path.exists(f"outdir/{stub}_buffer_categories.png")

    ds = buffer_categories(f'outdir/{stub}_cover_categories.tif', f'outdir/{stub}_gullies.tif', ridges_tif=None, roads_tif=None, outdir="outdir", stub=stub, buffer_width=3, savetif=True, plot=True)
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}  
    assert 'buffer_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_buffer_categories.tif")
    assert os.path.exists(f"outdir/{stub}_buffer_categories.png")


def test_shelter_metrics():
    """Simple shelter metrics tests, just checking the right files get created"""
    ds, df = patch_metrics(f"outdir/{stub}_buffer_categories.tif", outdir="outdir", stub=stub, plot=True, save_csv=True, save_tif=True)
    assert os.path.exists(f"outdir/{stub}_linear_categories.tif")
    assert os.path.exists(f"outdir/{stub}_linear_categories.png")
    assert os.path.exists(f"outdir/{stub}_labelled_categories.tif")
    assert os.path.exists(f"outdir/{stub}_labelled_categories.png")
    assert os.path.exists(f"outdir/{stub}_patch_metrics.csv")

    dfs = class_metrics(f"outdir/{stub}_linear_categories.tif", outdir="outdir", stub=stub, save_excel=True)
    assert os.path.exists(f"outdir/{stub}_class_metrics.xlsx")

if __name__ == '__main__':
    print("testing indices")
    start = time.time()

    test_basic()
    test_tree_categories()
    test_shelter_categories()
    test_cover_categories()
    test_buffer_categories()
    test_shelter_metrics()

    print(f"tests successfully completed in {time.time() - start} seconds")