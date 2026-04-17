"""
Test for indices_tif, indices_csv, and indices_tifs functions in all_indices.py
"""
import os
import numpy as np
import rioxarray as rxr

from shelterbelts.indices.all_indices import indices_tif, indices_tifs, indices_latlon

# Configuration
stub = 'g2_26729'
test_filename = f'data/{stub}_binary_tree_cover_10m.tiff'

# test_filename='/g/data/xe2/cb8590/Nick_Aus_treecover_10m/g1_02060_binary_tree_cover_10m.tiff'

def test_indices_tif():
    """Test basic execution of indices_tif with local data files"""
    result = indices_tif(
        test_filename,
        outdir='outdir',
        tmpdir='tmpdir',
        stub='test_pipeline'
    )

    # Function returns (ds_linear, df_patches)
    ds, df = result
    assert ds is not None, "indices_tif should return an xarray Dataset"
    assert df is not None, "indices_tif should return a pandas DataFrame"


def test_indices_tif_more_shelterbelts():
    """Test more-shelterbelts configuration against default, comparing each parameter's effect"""
    # Run default configuration
    indices_tif(test_filename, outdir='outdir', tmpdir='tmpdir', stub="default")

    # Run more-shelterbelts configuration
    indices_tif(
        test_filename,
        outdir='outdir',
        tmpdir='tmpdir',
        stub="more-shelterbelts",
        min_patch_size=5,  # Less scattered trees
        edge_size=5,  # Thicker core edges
        distance_threshold=30,  # Larger shelter radius
        density_threshold=3,  # Allows more sheltered areas
        buffer_width=5,  # Thicker riparian & road buffers
        min_shelterbelt_length=10,  # Allows shelterbelts to be shorter
    )

    # Load both outputs
    default_tif = os.path.join('outdir', 'default_linear_categories.tif')
    more_tif = os.path.join('outdir', 'more-shelterbelts_linear_categories.tif')
    assert os.path.exists(default_tif), f"Default output {default_tif} not created"
    assert os.path.exists(more_tif), f"More-shelterbelts output {more_tif} not created"

    da_default = rxr.open_rasterio(default_tif).isel(band=0).drop_vars('band')
    da_more = rxr.open_rasterio(more_tif).isel(band=0).drop_vars('band')

    # Get category counts for both
    default_cats = dict(zip(*np.unique(da_default.values, return_counts=True)))
    more_cats = dict(zip(*np.unique(da_more.values, return_counts=True)))

    # 1. min_patch_size=5 (Less scattered trees) - should have fewer scattered trees (11)
    default_scattered = default_cats.get(11, 0)
    more_scattered = more_cats.get(11, 0)
    assert more_scattered < default_scattered, \
        f"min_patch_size=5 should reduce scattered trees: default={default_scattered}, more={more_scattered}"

    # 2. edge_size=5 (Thicker core edges) - should have more patch edge (13)
    default_edge = default_cats.get(13, 0)
    more_edge = more_cats.get(13, 0)
    assert more_edge > default_edge, \
        f"edge_size=5 should increase patch edge: default={default_edge}, more={more_edge}"

    # 3 & 4. distance_threshold=30 + density_threshold=3 (Allows more sheltered areas) - should have more sheltered (32, 42)
    default_sheltered = default_cats.get(32, 0) + default_cats.get(42, 0)
    more_sheltered = more_cats.get(32, 0) + more_cats.get(42, 0)
    assert more_sheltered > default_sheltered, \
        f"distance_threshold=30 + density_threshold=3 should increase sheltered areas: default={default_sheltered}, more={more_sheltered}"

    # 5. buffer_width=5 (Thicker riparian & road buffers) - should have more buffer trees (15, 16, 17)
    default_buffers = default_cats.get(15, 0) + default_cats.get(16, 0) + default_cats.get(17, 0)
    more_buffers = more_cats.get(15, 0) + more_cats.get(16, 0) + more_cats.get(17, 0)
    assert more_buffers > default_buffers, \
        f"buffer_width=5 should increase buffer trees: default={default_buffers}, more={more_buffers}"

    # 6. min_shelterbelt_length=10 (Allows shelterbelts to be shorter) - should have more linear patches (18)
    default_linear = default_cats.get(18, 0)
    more_linear = more_cats.get(18, 0)
    assert more_linear > default_linear, \
        f"min_shelterbelt_length=10 should increase linear patches: default={default_linear}, more={more_linear}"


def test_indices_tif_less_shelterbelts():
    """Test less-shelterbelts configuration against default, comparing each parameter's effect"""
    # Run default configuration
    indices_tif(test_filename, outdir='outdir', tmpdir='tmpdir', stub="default")

    # Run less-shelterbelts configuration
    indices_tif(
        test_filename,
        outdir='outdir',
        tmpdir='tmpdir',
        stub="less-shelterbelts",
        min_patch_size=30,  # More scattered trees
        distance_threshold=10,  # Smaller shelter radius
        density_threshold=10,  # Forces less sheltered areas
        buffer_width=2,  # Thinner riparian & road buffers
        strict_core_area=False,  # More core areas
        max_shelterbelt_width=4,  # Forces shelterbelts to be thinner
    )

    # Load both outputs
    default_tif = os.path.join('outdir', 'default_linear_categories.tif')
    less_tif = os.path.join('outdir', 'less-shelterbelts_linear_categories.tif')
    assert os.path.exists(default_tif), f"Default output {default_tif} not created"
    assert os.path.exists(less_tif), f"Less-shelterbelts output {less_tif} not created"

    da_default = rxr.open_rasterio(default_tif).isel(band=0).drop_vars('band')
    da_less = rxr.open_rasterio(less_tif).isel(band=0).drop_vars('band')

    # Get category counts for both
    default_cats = dict(zip(*np.unique(da_default.values, return_counts=True)))
    less_cats = dict(zip(*np.unique(da_less.values, return_counts=True)))

    # 1. min_patch_size=30 (More scattered trees) - should have more scattered trees (11)
    default_scattered = default_cats.get(11, 0)
    less_scattered = less_cats.get(11, 0)
    assert less_scattered > default_scattered, \
        f"min_patch_size=30 should increase scattered trees: default={default_scattered}, less={less_scattered}"

    # 2. distance_threshold=10 + density_threshold=10 (Forces less sheltered areas) - should have fewer sheltered (32, 42)
    default_sheltered = default_cats.get(32, 0) + default_cats.get(42, 0)
    less_sheltered = less_cats.get(32, 0) + less_cats.get(42, 0)
    assert less_sheltered < default_sheltered, \
        f"distance_threshold=10 + density_threshold=10 should decrease sheltered areas: default={default_sheltered}, less={less_sheltered}"

    # 3. buffer_width=2 (Thinner riparian & road buffers) - should have fewer buffer trees (15, 16, 17)
    default_buffers = default_cats.get(15, 0) + default_cats.get(16, 0) + default_cats.get(17, 0)
    less_buffers = less_cats.get(15, 0) + less_cats.get(16, 0) + less_cats.get(17, 0)
    assert less_buffers < default_buffers, \
        f"buffer_width=2 should decrease buffer trees: default={default_buffers}, less={less_buffers}"

    # 4. max_shelterbelt_width=4 (Forces shelterbelts to be thinner) - should have fewer/different linear patches (18, 19)
    default_patches = default_cats.get(18, 0) + default_cats.get(19, 0)
    less_patches = less_cats.get(18, 0) + less_cats.get(19, 0)
    # With thinner shelterbelts, some may get reclassified to non-linear, so patches may be different
    assert (less_patches != default_patches), \
        f"max_shelterbelt_width=4 should affect linear/non-linear classification: default={default_patches}, less={less_patches}"


def test_indices_latlon():
    """Smoke test for indices_latlon using the ACT test area."""
    ds, df = indices_latlon(
        -34.389, 148.469, buffer=0.02,
        outdir='outdir', tmpdir='tmpdir', stub='test_latlon'
    )
    assert ds is not None
    assert 'linear_categories' in ds.data_vars
    assert df is not None


def test_indices_tifs():
    "Check all_indices works on a folder of tifs."
    import shelterbelts.indices.all_indices as all_indices_module
    script_dir = os.path.dirname(os.path.abspath(all_indices_module.__file__))
    folder = os.path.join(script_dir, 'data', 'multiple_binary_tifs')
    indices_tifs(folder, suffix='tiff')

    # why did this work on one the first time (not both), but is now starting with 0 percent_tifs?
