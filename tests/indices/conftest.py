import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_dirs():
    """Create necessary directories for tests"""
    if not os.path.exists('tmpdir'):
        os.mkdir('tmpdir')
    if not os.path.exists('outdir'):
        os.mkdir('outdir')


# Configuration
stub = 'g2_26729'
test_filename = f'data/{stub}_binary_tree_cover_10m.tiff'
