"""Tests for the OpenStreetMap roads API."""

import os

from shelterbelts.apis.osm import osm_roads

# The ACT test area, which test_indices_latlon also uses. Roads cross it.
test_filename = 'data/g2_26729_binary_tree_cover_10m.tiff'


def test_osm_roads_basic():
    """Test osm_roads download and output structure."""
    gdf, ds = osm_roads(test_filename, outdir='outdir', stub='test_osm')
    assert 'roads' in ds.data_vars
    assert os.path.exists("outdir/test_osm_roads.tif")
    assert os.path.exists("outdir/test_osm_roads.gpkg")
    assert len(gdf) > 0, "expected at least one road in the ACT test area"
