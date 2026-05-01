import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_dirs():
    """Create necessary directories for tests."""
    for d in ('tmpdir', 'outdir'):
        if not os.path.exists(d):
            os.mkdir(d)
