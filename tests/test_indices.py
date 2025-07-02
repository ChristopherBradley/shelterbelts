import os
import time

from shelterbelts.indices.tree_categories import tree_categories

if not os.path.exists('tmpdir'):
    os.mkdir('tmpdir')
if not os.path.exists('outdir'):
    os.mkdir('outdir')

def test_basic():
    """Basic tests for each of the files"""
    ds = tree_categories('TEST.tif', min_length=10, edge_size=3, gap_size=1, ds=None, save_tif=True)
    assert set(ds.coords) == {'latitude', 'longitude'}  
    assert set(ds.data_vars) == {'tree_categories'}
    assert os.path.exists("outdir/TEST_categorised.tif")