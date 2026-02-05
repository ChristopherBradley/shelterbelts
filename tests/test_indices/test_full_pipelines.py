"""
Test for run_pipeline_tif function in full_pipelines.py
"""
import os
import tempfile

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
            stub='test_pipeline',
            wind_method=None
        )
        
        # Function returns None but should complete without errors
        assert result is None, "run_pipeline_tif should return None"


# --max_shelterbelt_width 8 --min_shelterbelt_length 10 --min_core_size 10000 --edge_size 5 --buffer_width 5 --distance_threshold 30 --density_threshold 3 --min_patch_size 10 --max_gap_size 2
