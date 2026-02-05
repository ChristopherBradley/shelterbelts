"""Tests for BARRA daily API."""

import os
from shelterbelts.apis.barra_daily import barra_daily


def test_barra_daily_basic():
    """Test BARRA daily download and output structure."""
    ds = barra_daily(lat=-34.37825, lon=148.42490, buffer=0.1, start_year=2020, end_year=2020, outdir='outdir', stub='test_barra')
    assert 'uas' in ds.data_vars
    assert 'vas' in ds.data_vars
    assert os.path.exists("outdir/test_barra_barra_daily.nc")


def test_barra_daily_multi_year():
    """Test BARRA daily with multiple years."""
    ds = barra_daily(lat=-34.37825, lon=148.42490, buffer=0.1, start_year=2020, end_year=2021, outdir='outdir', stub='test_barra_multi')
    assert 'uas' in ds.data_vars
    assert 'vas' in ds.data_vars
    assert ds.sizes['time'] > 365  # Should span multiple years
