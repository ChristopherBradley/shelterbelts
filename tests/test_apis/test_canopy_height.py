"""Tests for canopy height API."""

import os
from shelterbelts.apis.canopy_height import canopy_height


def test_canopy_height_basic():
    """Test canopy height download and output structure."""
    ds = canopy_height(lat=-34.37825, lon=148.42490, buffer=0.012, outdir='outdir', stub='test_ch', tmpdir='tmpdir')
    assert 'canopy_height' in ds.data_vars
    assert os.path.exists("outdir/test_ch_canopy_height.tif")
