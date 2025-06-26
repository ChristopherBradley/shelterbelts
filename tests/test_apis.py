
import os
import time

from shelterbelts.apis.worldcover import worldcover
from shelterbelts.apis.canopy_height import canopy_height


if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')


def test_basic():
    """Basic tests for each of the files"""
    ds = worldcover(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST')
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'worldcover'}
    assert os.path.exists("outdir/TEST_worldcover.tif")
    assert os.path.exists("outdir/TEST_worldcover.png")

    ds = canopy_height(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST', tmpdir='tmpdir')
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'canopy_height'}
    assert os.path.exists("outdir/TEST_canopy_height.tif")
    assert os.path.exists("outdir/TEST_canopy_height.png")


def test_canopy_height():
    """More comprehensive Tolan Global Canopy Height tests: 2 buffers, with and without savetif and plotting"""
    ds = canopy_height(lat=-34.389, lon=148.469, buffer=0, outdir='outdir', stub='TEST', tmpdir='tmpdir')
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'canopy_height'}

    if os.path.exists("outdir/TEST_canopy_height.tif"):
        os.remove("outdir/TEST_canopy_height.tif")
    ds = canopy_height(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST', tmpdir='tmpdir', save_tif=False)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'canopy_height'}
    assert not os.path.exists("outdir/TEST_canopy_height.tif")

    if os.path.exists("outdir/TEST_canopy_height.png"):
        os.remove("outdir/TEST_canopy_height.png")
    ds = canopy_height(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST', tmpdir='tmpdir', plot=False)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'canopy_height'}
    assert not os.path.exists("outdir/TEST_canopy_height.png")


def test_worldcover():
    """More comprehensive worldcover tests: 2 buffers, with and without savetif and plotting"""
    ds = worldcover(lat=-34.389, lon=148.469, buffer=0, outdir='outdir', stub='TEST')
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'worldcover'}

    if os.path.exists("outdir/TEST_worldcover.tif"):
        os.remove("outdir/TEST_worldcover.tif")
    ds = worldcover(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST', save_tif=False)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'worldcover'}
    assert not os.path.exists("outdir/TEST_worldcover.tif")

    if os.path.exists("outdir/TEST_worldcover.png"):
        os.remove("outdir/TEST_worldcover.png")
    ds = worldcover(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST', plot=False)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'worldcover'}
    assert not os.path.exists("outdir/TEST_worldcover.png")

if __name__ == '__main__':
    print("Testing APIs")
    start = time.time()

    test_basic()
    test_worldcover()
    test_canopy_height()
    
    print(f"Tests successfully completed in {time.time() - start} seconds")