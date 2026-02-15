"""Tests for catchments function."""

from shelterbelts.indices.catchments import catchments

# Ran these commands in the terminal to get rid of postgres errors (they didn't prevent the code from running, just clogged up the outputs).
# rm $CONDA_PREFIX/lib/gdalplugins/ogr_PG.dylib
# rm $CONDA_PREFIX/lib/gdalplugins/gdal_PostGISRaster.dylib
# export KMP_WARNINGS=0 (put this in my .zshrc)

def test_catchments_basic():
    """Test catchments function returns rasterized output."""
    geotif = 'outdir/g2_26729_tree_categories.tif'  # TODO: This should be a DEM instead.
    ds = catchments(geotif, outdir='outdir', stub='test_catch', savetif=False, plot=False)
    assert all(var in ds.data_vars for var in ['terrain', 'gullies', 'ridges'])

    # TODO: Should assert that the values in the array aren't all the same
