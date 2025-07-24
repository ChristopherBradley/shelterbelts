import os
import time

from shelterbelts.indices.tree_categories import tree_categories
from shelterbelts.indices.shelter_categories import shelter_categories
from shelterbelts.indices.cover_categories import cover_categories
from shelterbelts.indices.hydrolines import hydrolines

if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')

# Should add this to git so the tests can run after doing a git clone
stub = 'g2_26729'
test_filename = f'data/{stub}_binary_tree_cover_10m.tiff'


def test_basic():
    """Basic tests for each of the files"""
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert os.path.exists(f"outdir/{stub}_categorised.tif")
    assert os.path.exists(f"outdir/{stub}_categorised.png")

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=stub, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_shelter_categories.tif")
    assert os.path.exists(f"outdir/{stub}_shelter_categories.png")

def test_cover_categories():
    """More comprehensive cover category tests: with and without saving an plotting"""
    ds = cover_categories(f"outdir/{stub}_shelter_categories.tif", f"outdir/{stub}_worldcover.tif", outdir='outdir', stub=stub, savetif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'cover_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_cover_categories.tif")
    assert os.path.exists(f"outdir/{stub}_cover_categories.png")


def test_shelter_categories():
    """More comprehensive shelter category tests: with & without height tif & wind ds, 3x wind_methods, 2x wind_threshold, density_threshold, distance_threshold, minimum_height, with and without savetif and plotting"""
    # Changing the minimum height
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_minCH1', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_minCH5', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=5, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_minCH10', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the wind method
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodMAX', wind_method='MAX', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodANY', wind_method='ANY', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodCOMMON', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_methodHAPPENED', wind_method='HAPPENED', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the wind speed threshold
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w10', wind_method='HAPPENED', wind_threshold=10, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w20', wind_method='HAPPENED', wind_threshold=20, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w25', wind_method='HAPPENED', wind_threshold=25, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_w30', wind_method='HAPPENED', wind_threshold=30, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the distance threshold
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_d30', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=30, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_d15', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=15, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=None, outdir='outdir', stub=f'{stub}_d10', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=10, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the distance threshold, with a corresponding height tif
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_dCH30', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=30, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_dCH15', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=15, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=f"outdir/{stub}_barra_daily.nc", height_tif=f"outdir/{stub}_canopy_height.tif", outdir='outdir', stub=f'{stub}_dCH10', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=10, density_threshold=10, minimum_height=1, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    # Changing the density threshold
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=f'{stub}_density5', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=5, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=f'{stub}_density20', wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=20, minimum_height=10, savetif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert 'shelter_categories' in set(ds.data_vars)

    if os.path.exists(f"outdir/{stub}_shelter_categories.tif"):
        os.remove(f"outdir/{stub}_shelter_categories.tif")
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=stub, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=False, plot=True)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.tif")

    if os.path.exists(f"outdir/{stub}_shelter_categories.png"):
        os.remove(f"outdir/{stub}_shelter_categories.png")
    ds = shelter_categories(f"outdir/{stub}_categorised.tif", wind_ds=None, height_tif=None, outdir='outdir', stub=stub, wind_method='MOST_COMMON', wind_threshold=15, distance_threshold=20, density_threshold=10, minimum_height=10, savetif=True, plot=False)
    assert not os.path.exists(f"outdir/{stub}_shelter_categories.png")


def test_tree_categories():
    """More comprehensive tree category tests: 2x patch sizes, 2x edge sizes, 2x max_gap_sizes, without saving tif, without plot"""
    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_patch50', min_patch_size=50, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_edge10', min_patch_size=10, edge_size=10, max_gap_size=2, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    ds = tree_categories(test_filename, outdir='outdir', stub=f'{stub}_gap0', min_patch_size=10, edge_size=3, max_gap_size=0, ds=None, save_tif=True, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}

    if os.path.exists(f"outdir/{stub}_categorised.tif"):
        os.remove(f"outdir/{stub}_categorised.tif")
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=False, plot=True)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_categorised.tif")

    if os.path.exists(f"outdir/{stub}_categorised.png"):
        os.remove(f"outdir/{stub}_categorised.png")
    ds = tree_categories(test_filename, outdir='outdir', stub=stub, min_patch_size=10, edge_size=3, max_gap_size=2, ds=None, save_tif=True, plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'woody_veg', 'tree_categories'}
    assert not os.path.exists(f"outdir/{stub}_categorised.png")

def test_hydrolines():
# Leaving these tests out of the pipeline because it takes so long to read in the hydrolines file
    hydrolines_gdb = "/Users/christopherbradley/Documents/PHD/Data/Australia_datasets/SurfaceHydrologyLinesRegional.gdb"
    outdir = 'outdir/'
    stub = 'g2_26729'
    geotif = f"{outdir}{stub}_categorised.tif"
    ds = hydrolines(geotif, hydrolines_gdb, outdir=outdir, stub=stub)

if __name__ == '__main__':
    print("testing indices")
    start = time.time()

    # test_basic()
    # test_tree_categories()
    # test_shelter_categories()
    # test_cover_categories()
    test_hydrolines()

    print(f"tests successfully completed in {time.time() - start} seconds")