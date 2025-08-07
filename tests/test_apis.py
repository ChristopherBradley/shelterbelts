
import os
import time

from shelterbelts.apis.worldcover import worldcover
from shelterbelts.apis.canopy_height import canopy_height
from shelterbelts.apis.barra_daily import barra_daily
from shelterbelts.apis.hydrolines import hydrolines


if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')


def test_basic():
    """Basic tests for each of the files"""
    ds = worldcover(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='g2_26729')
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'worldcover'}
    assert os.path.exists("outdir/g2_26729_worldcover.tif")
    assert os.path.exists("outdir/g2_26729_worldcover.png")

    ds = canopy_height(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='g2_26729', tmpdir='tmpdir')
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'canopy_height'}
    assert os.path.exists("outdir/g2_26729_canopy_height.tif")
    assert os.path.exists("outdir/g2_26729_canopy_height.png")

    ds = barra_daily(lat=-34.37825, lon=148.42490, buffer=0.1, start_year=2020, end_year=2021, outdir='outdir', stub='g2_26729')
    assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'uas', 'vas'}
    assert os.path.exists("outdir/g2_26729_barra_daily.nc")
    assert os.path.exists("outdir/g2_26729_barra_daily.png")
    assert ds.sizes['latitude'] >= 1 and ds.sizes['longitude'] >= 1

def test_barra_daily():
    """More comprehensive barra download tests: 2 buffers, single or multiple years, with and without save netcdf and plotting"""
    ds = barra_daily(lat=-34.37825, lon=148.42490, buffer=0.1, start_year=2020, end_year=2020, outdir='outdir', stub='g2_26729')
    assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'uas', 'vas'}
    assert os.path.exists("outdir/g2_26729_barra_daily.nc")
    assert os.path.exists("outdir/g2_26729_barra_daily.png")
    assert ds.sizes['latitude'] >= 1 and ds.sizes['longitude'] >= 1

    ds = barra_daily(lat=-34.37825, lon=148.42490, buffer=0, start_year=2020, end_year=2020, outdir='outdir', stub='g2_26729')
    assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'uas', 'vas'}
    assert os.path.exists("outdir/g2_26729_barra_daily.nc")
    assert os.path.exists("outdir/g2_26729_barra_daily.png")
    assert ds.sizes['latitude'] >= 1 and ds.sizes['longitude'] >= 1

    ds = barra_daily(lat=-34.37825, lon=148.42490, buffer=0.1, start_year=2020, end_year=2020, outdir='outdir', stub='g2_26729', save_netcdf=False)
    assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'uas', 'vas'}
    assert os.path.exists("outdir/g2_26729_barra_daily.nc")
    assert os.path.exists("outdir/g2_26729_barra_daily.png")
    assert ds.sizes['latitude'] >= 1 and ds.sizes['longitude'] >= 1

    ds = barra_daily(lat=-34.37825, lon=148.42490, buffer=0.1, start_year=2020, end_year=2020, outdir='outdir', stub='g2_26729', plot=False)
    assert set(ds.coords) == {'time', 'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'uas', 'vas'}
    assert os.path.exists("outdir/g2_26729_barra_daily.nc")
    assert os.path.exists("outdir/g2_26729_barra_daily.png")
    assert ds.sizes['latitude'] >= 1 and ds.sizes['longitude'] >= 1


def test_canopy_height():
    """More comprehensive Tolan Global Canopy Height tests: 2 buffers, with and without savetif and plotting"""
    ds = canopy_height(lat=-34.37825, lon=148.42490, buffer=0, outdir='outdir', stub='g2_26729', tmpdir='tmpdir')
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'canopy_height'}

    if os.path.exists("outdir/g2_26729_canopy_height.tif"):
        os.remove("outdir/g2_26729_canopy_height.tif")
    ds = canopy_height(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='g2_26729', tmpdir='tmpdir', save_tif=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'canopy_height'}
    assert not os.path.exists("outdir/g2_26729_canopy_height.tif")

    if os.path.exists("outdir/g2_26729_canopy_height.png"):
        os.remove("outdir/g2_26729_canopy_height.png")
    ds = canopy_height(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='g2_26729', tmpdir='tmpdir', plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'canopy_height'}
    assert not os.path.exists("outdir/g2_26729_canopy_height.png")


def test_worldcover():
    """More comprehensive worldcover tests: 2 buffers, with and without savetif and plotting"""
    ds = worldcover(lat=-34.37825, lon=148.42490, buffer=0, outdir='outdir', stub='g2_26729')
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'worldcover'}

    if os.path.exists("outdir/g2_26729_worldcover.tif"):
        os.remove("outdir/g2_26729_worldcover.tif")
    ds = worldcover(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='g2_26729', save_tif=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'worldcover'}
    assert not os.path.exists("outdir/g2_26729_worldcover.tif")

    if os.path.exists("outdir/g2_26729_worldcover.png"):
        os.remove("outdir/g2_26729_worldcover.png")
    ds = worldcover(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='g2_26729', plot=False)
    assert set(ds.coords) == {'latitude', 'longitude', 'spatial_ref'}  
    assert set(ds.data_vars) == {'worldcover'}
    assert not os.path.exists("outdir/g2_26729_worldcover.png")


def test_hydrolines():
    print()
    hydrolines_gpkg = "data/g2_26729_hydrolines_cropped.gpkg"  
    stub = 'g2_26729'
    outdir = 'outdir'
    geotif = os.path.join(outdir, f"{stub}_categorised.tif")
    ds = hydrolines(geotif, hydrolines_gpkg, outdir=outdir, stub=stub)

def test_catchments():
    print()

def test_roads():
    print()

if __name__ == '__main__':
    print("testing APIs")
    start = time.time()

    test_basic()
    test_worldcover()
    test_canopy_height()
    test_barra_daily()

    test_hydrolines()

    
    print(f"tests successfully completed in {time.time() - start} seconds")