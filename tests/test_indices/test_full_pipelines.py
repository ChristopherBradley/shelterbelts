"""
Test for run_pipeline_tif function in full_pipelines.py
"""
import os
import tempfile
import numpy as np
import rioxarray as rxr

from shelterbelts.indices.full_pipelines import run_pipeline_tif

# Configuration
stub = 'g2_26729'
test_filename = f'data/{stub}_binary_tree_cover_10m.tiff'


def test_run_pipeline_tif_basic():
    """Test basic execution of run_pipeline_tif with local data files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, 'output')
        os.makedirs(outdir, exist_ok=True)
        
        result = run_pipeline_tif(
            test_filename,
            outdir=outdir,
            tmpdir=tmpdir,
            stub='test_pipeline'
        )
        
        # Function returns None but should complete without errors
        assert result is None, "run_pipeline_tif should return None"


def test_run_pipeline_tif_more_shelterbelts():
    """Test more-shelterbelts configuration against default, comparing each parameter's effect"""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, 'output')
        os.makedirs(outdir, exist_ok=True)
        
        # Run default configuration
        run_pipeline_tif(
            test_filename,
            outdir=outdir,
            tmpdir=tmpdir,
            stub="default"
        )
        
        # Run more-shelterbelts configuration
        run_pipeline_tif(
            test_filename,
            outdir=outdir,
            tmpdir=tmpdir,
            stub="more-shelterbelts",
            min_patch_size=5,  # Less scattered trees
            edge_size=5,  # Thicker core edges
            distance_threshold=30,  # Larger shelter radius
            density_threshold=3,  # Allows more sheltered areas
            buffer_width=5,  # Thicker riparian & road buffers
            min_shelterbelt_length=10,  # Allows shelterbelts to be shorter
        )
        
        # Load both outputs
        default_tif = os.path.join(outdir, 'default_linear_categories.tif')
        more_tif = os.path.join(outdir, 'more-shelterbelts_linear_categories.tif')
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


def test_run_pipeline_tif_less_shelterbelts():
    """Test less-shelterbelts configuration against default, comparing each parameter's effect"""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, 'output')
        os.makedirs(outdir, exist_ok=True)
        
        # Run default configuration
        run_pipeline_tif(
            test_filename,
            outdir=outdir,
            tmpdir=tmpdir,
            stub="default"
        )
        
        # Run less-shelterbelts configuration
        run_pipeline_tif(
            test_filename,
            outdir=outdir,
            tmpdir=tmpdir,
            stub="less-shelterbelts",
            min_patch_size=30,  # More scattered trees
            distance_threshold=10,  # Smaller shelter radius
            density_threshold=10,  # Forces less sheltered areas
            buffer_width=2,  # Thinner riparian & road buffers
            strict_core_area=False,  # More core areas
            max_shelterbelt_width=4,  # Forces shelterbelts to be thinner
        )
        
        # Load both outputs
        default_tif = os.path.join(outdir, 'default_linear_categories.tif')
        less_tif = os.path.join(outdir, 'less-shelterbelts_linear_categories.tif')
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
