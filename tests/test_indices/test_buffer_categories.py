import os

from shelterbelts.indices.buffer_categories import buffer_categories


stub = 'g2_26729'
cover_categories_file = f'data/{stub}_cover_categories.tif'
gullies_file = f'data/{stub}_DEM-S_gullies.tif'
ridges_file = f'data/{stub}_DEM-S_ridges.tif'
roads_file = f'data/{stub}_roads.tif'


def test_buffer_categories_basic():
    """Basic test for buffer_categories function"""
    ds = buffer_categories(
        cover_categories_file,
        gullies_file,
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
        cover_categories_file,
        gullies_file,
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_ridges():
    """Test buffer_categories with ridges tif"""
    ds = buffer_categories(
        cover_categories_file,
        gullies_file,
        ridges_data=ridges_file,
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_roads():
    """Test buffer_categories with roads tif"""
    ds = buffer_categories(
        cover_categories_file,
        gullies_file,
        roads_data=roads_file,
        outdir="outdir",
        stub=stub,
        buffer_width=5
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'buffer_categories' in set(ds.data_vars)


def test_buffer_categories_with_ridges_and_roads():
    """Test buffer_categories with both ridges and roads tif"""
    ds = buffer_categories(
        cover_categories_file,
        gullies_file,
        ridges_data=ridges_file,
        roads_data=roads_file,
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
        cover_categories_file,
        gullies_file,
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
        cover_categories_file,
        gullies_file,
        outdir="outdir",
        stub=stub,
        plot=False
    )
    assert not os.path.exists(f"outdir/{stub}_buffer_categories.png")
