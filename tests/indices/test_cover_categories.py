import os

from shelterbelts.indices import cover_categories


stub = 'g2_26729'


def test_cover_categories_basic():
    """Basic test for cover_categories function"""
    ds = cover_categories(
        f"outdir/{stub}_shelter_categories.tif",
        f"outdir/{stub}_worldcover.tif",
        outdir='outdir',
        stub=stub
    )
    assert set(ds.coords) == {'x', 'y', 'spatial_ref'}
    assert 'cover_categories' in set(ds.data_vars)
    assert os.path.exists(f"outdir/{stub}_cover_categories.tif")
    assert os.path.exists(f"outdir/{stub}_cover_categories.png")


def test_cover_categories_no_save():
    """Test cover_categories without saving tif"""
    if os.path.exists(f"outdir/{stub}_cover_categories.tif"):
        os.remove(f"outdir/{stub}_cover_categories.tif")
    
    ds = cover_categories(
        f"outdir/{stub}_shelter_categories.tif",
        f"outdir/{stub}_worldcover.tif",
        outdir='outdir',
        stub=stub,
        savetif=False
    )
    assert not os.path.exists(f"outdir/{stub}_cover_categories.tif")


def test_cover_categories_no_plot():
    """Test cover_categories without plotting"""
    if os.path.exists(f"outdir/{stub}_cover_categories.png"):
        os.remove(f"outdir/{stub}_cover_categories.png")
    
    ds = cover_categories(
        f"outdir/{stub}_shelter_categories.tif",
        f"outdir/{stub}_worldcover.tif",
        outdir='outdir',
        stub=stub,
        plot=False
    )
    assert not os.path.exists(f"outdir/{stub}_cover_categories.png")
