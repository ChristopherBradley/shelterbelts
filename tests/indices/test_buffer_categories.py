import os
import pytest

from shelterbelts.indices.buffer_categories import buffer_categories


stub = 'g2_26729'


def test_buffer_categories_basic():
    """Basic test for buffer_categories function"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_buffer_categories.tif")
    assert os.path.exists(f"outdir/{stub}_buffer_categories.png")


def test_buffer_categories_buffer_width():
    """Test buffer_categories with different buffer widths"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_ridges():
    """Test buffer_categories with ridges tif"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        ridges_tif=f'outdir/{stub}_ridges.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_roads():
    """Test buffer_categories with roads tif"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        roads_tif=f'outdir/{stub}_roads.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_ridges_and_roads():
    """Test buffer_categories with both ridges and roads tif"""
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        ridges_tif=f'outdir/{stub}_ridges.tif',
        roads_tif=f'outdir/{stub}_roads.tif',
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_no_save():
    """Test buffer_categories without saving tif"""
    if os.path.exists(f"outdir/{stub}_buffer_categories.tif"):
        os.remove(f"outdir/{stub}_buffer_categories.tif")
    
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub,
        savetif=False
    )
    assert not os.path.exists(f"outdir/{stub}_buffer_categories.tif")


def test_buffer_categories_no_plot():
    """Test buffer_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_buffer_categories.png"):
        os.remove(f"outdir/{stub}_buffer_categories.png")
    
    ds = buffer_categories(
        f'outdir/{stub}_cover_categories.tif',
        f'outdir/{stub}_gullies.tif',
        outdir="outdir",
        stub=stub,
        plot=False
    )
    assert not os.path.exists(f"outdir/{stub}_buffer_categories.png")
