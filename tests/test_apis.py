
import os

from shelterbelts.apis.worldcover import worldcover


if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')


# Basic tests
ds = worldcover(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST')
assert set(ds.coords) == {'latitude', 'longitude'}  
assert set(ds.data_vars) == {'worldcover'}
assert os.path.exists("outdir/TEST_worldcover.tif")
assert os.path.exists("outdir/TEST_worldcover.png")



# More comprehensive worldcover tests: 2 buffers, with and without savetif and plotting
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