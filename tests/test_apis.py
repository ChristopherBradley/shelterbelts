
import os

from shelterbelts.apis.worldcover import worldcover


if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')


# WorldCover tests: 2 buffers, with and without savetif and plotting
ds = worldcover(lat=-34.389, lon=148.469, buffer=0.01, outdir='outdir', stub='TEST')
assert set(ds.coords) == {'latitude', 'longitude'}  
assert set(ds.data_vars) == {'worldcover'}
assert os.path.exists("outdir/TEST_worldcover.tif")
assert os.path.exists("outdir/TEST_worldcover.png")

