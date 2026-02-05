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
            wind_method=None,
            worldcover_dir='data',
            worldcover_geojson='g2_26729_worldcover_footprints.geojson',
            hydrolines_gdb='data/g2_26729_hydrolines_cropped.gpkg',
            roads_gdb='data/g2_26729_roads_cropped.gpkg'
        )
        
        # Function returns None but should complete without errors
        assert result is None, "run_pipeline_tif should return None"
