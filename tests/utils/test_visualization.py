"""Tests for visualization utilities."""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pytest

from shelterbelts.utils.visualization import visualise_categories
from shelterbelts.indices.tree_categories import tree_categories_cmap, tree_categories_labels


@pytest.fixture
def sample_categorical_data():
    """Create sample 3x3 array with all categories."""
    data = np.array([
        [ 0, 11, 12],
        [13, 14,  0],
        [11, 12, 13]
    ], dtype=np.uint8)
    return xr.DataArray(data, dims=['y', 'x'])


def test_visualise_categories_basic(sample_categorical_data):
    """Test that visualise_categories runs without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.png")
        visualise_categories(
            sample_categorical_data,
            filename=filepath,
            colormap=tree_categories_cmap,
            labels=tree_categories_labels
        )
        assert os.path.exists(filepath)


def test_visualise_categories_colors(sample_categorical_data):
    """Test that each category appears with its expected color."""
    from PIL import Image
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.png")
        visualise_categories(sample_categorical_data, filepath, tree_categories_cmap, tree_categories_labels)
        
        img = np.array(Image.open(filepath).convert('RGB'))
        
        for category in [0, 11, 12, 13, 14]:
            expected = np.array(tree_categories_cmap[category])
            # Check color appears somewhere in image (within tolerance for rendering)
            matches = np.all(np.abs(img - expected) < 20, axis=2)
            assert matches.any(), f"Category {category} color not found"


def test_visualise_categories_sidebyside(sample_categorical_data):
    """Test side-by-side visualization."""
    from shelterbelts.utils.visualization import visualise_categories_sidebyside
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_sidebyside.png")
        visualise_categories_sidebyside(
            sample_categorical_data,
            sample_categorical_data,
            filename=filepath,
            colormap=tree_categories_cmap,
            labels=tree_categories_labels,
            title1="Left",
            title2="Right"
        )
        assert os.path.exists(filepath)
