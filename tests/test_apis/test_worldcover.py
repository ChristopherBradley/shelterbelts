"""Tests for worldcover API."""

import os
from shelterbelts.apis.worldcover import worldcover


def test_worldcover_basic():
    """Test worldcover download and output structure."""
    ds = worldcover(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='test_wc')
    assert 'worldcover' in ds.data_vars
    assert os.path.exists("outdir/test_wc_worldcover.tif")


def test_worldcover_doctest():
    """Test worldcover doctest example: small tile without saving files."""
    ds = worldcover(buffer=0.01, save_tif=False, plot=False)
    assert 'worldcover' in ds.data_vars
