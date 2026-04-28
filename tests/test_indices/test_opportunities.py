import numpy as np
import rioxarray as rxr

from shelterbelts.indices.opportunities import opportunities_da, opportunities


stub = 'g2_26729'
tree_file = f'data/{stub}_binary_tree_cover_10m.tiff'
roads_file = f'data/{stub}_roads.tif'
gullies_file = f'data/{stub}_hydrolines.tif'
dem_file = f'data/{stub}_DEM-H.tif'
worldcover_file = f'data/{stub}_worldcover.tif'


def _load_aligned():
    """Load real data files and align to the same grid."""
    da_trees = rxr.open_rasterio(tree_file).isel(band=0).drop_vars('band')
    da_roads = rxr.open_rasterio(roads_file).isel(band=0).drop_vars('band')
    da_gullies = rxr.open_rasterio(gullies_file).isel(band=0).drop_vars('band')
    da_dem = rxr.open_rasterio(dem_file).isel(band=0).drop_vars('band')
    da_worldcover = rxr.open_rasterio(worldcover_file).isel(band=0).drop_vars('band')
    da_worldcover = da_worldcover.rio.reproject_match(da_trees)
    return da_trees, da_roads, da_gullies, da_dem, da_worldcover


def test_opportunities_da_basic():
    """Basic test for opportunities_da using real data."""
    da_trees, da_roads, da_gullies, da_dem, da_worldcover = _load_aligned()

    ds = opportunities_da(
        da_trees, da_roads, da_gullies, None, da_dem, da_worldcover,
        outdir='outdir', stub='test', width=3, savetif=False, plot=False, crop_pixels=0,
    )

    assert set(ds.data_vars) == {'woody_veg', 'opportunities'}
    unique_vals = set(np.unique(ds['opportunities'].values))
    assert unique_vals.issubset({0, 5, 6, 7, 8})


def test_opportunities_da_width():
    """Test that changing width affects the buffer size."""
    da_trees, da_roads, da_gullies, da_dem, da_worldcover = _load_aligned()

    ds_w1 = opportunities_da(
        da_trees, da_roads, da_gullies, None, da_dem, da_worldcover,
        savetif=False, plot=False, width=1, crop_pixels=0,
    )
    ds_w3 = opportunities_da(
        da_trees, da_roads, da_gullies, None, da_dem, da_worldcover,
        savetif=False, plot=False, width=3, crop_pixels=0,
    )

    n_w1 = (ds_w1['opportunities'].values > 0).sum()
    n_w3 = (ds_w3['opportunities'].values > 0).sum()
    assert n_w3 >= n_w1


def test_opportunities_da_trees_exclude():
    """Existing trees should be excluded from opportunities."""
    da_trees, da_roads, da_gullies, da_dem, da_worldcover = _load_aligned()

    ds = opportunities_da(
        da_trees, da_roads, da_gullies, None, da_dem, da_worldcover,
        savetif=False, plot=False, width=3, crop_pixels=0,
    )

    # Where there are trees, there should be no opportunities
    tree_mask = da_trees.values.astype(bool)
    assert (ds['opportunities'].values[tree_mask] > 0).sum() == 0


def test_opportunities_da_worldcover_filter():
    """Only grass (30) and cropland (40) should produce opportunities."""
    da_trees, da_roads, da_gullies, da_dem, da_worldcover = _load_aligned()

    # Set worldcover to urban (50) — nothing should be flagged
    da_worldcover_urban = da_worldcover.copy()
    da_worldcover_urban.values[:] = 50

    ds = opportunities_da(
        da_trees, da_roads, da_gullies, None, da_dem, da_worldcover_urban,
        savetif=False, plot=False, width=3, crop_pixels=0,
    )

    assert (ds['opportunities'].values > 0).sum() == 0


def test_opportunities_with_file_paths():
    """Test opportunities() with real data file paths."""
    ds = opportunities(
        tree_file, outdir='outdir', stub='test_files', savetif=False, plot=False,
        roads_data=roads_file, gullies_data=gullies_file, dem_data=dem_file, worldcover_data=worldcover_file,
    )

    assert set(ds.data_vars) == {'woody_veg', 'opportunities'}
    unique_vals = set(np.unique(ds['opportunities'].values))
    assert unique_vals.issubset({0, 5, 6, 7, 8})


def test_opportunities_does_not_mutate_input():
    """Ensure input DataArrays are not mutated in-place."""
    da_trees, da_roads, da_gullies, da_dem, da_worldcover = _load_aligned()
    trees_before = da_trees.copy(deep=True)

    opportunities_da(
        da_trees, da_roads, da_gullies, None, da_dem, da_worldcover,
        savetif=False, plot=False, width=1, crop_pixels=0,
    )

    assert da_trees.equals(trees_before)


def test_opportunities_crop_and_rasterize():
    """Test that opportunities() uses crop_and_rasterize when roads_data and gullies_data are None."""
    ds = opportunities(
        tree_file, dem_data=dem_file, worldcover_data=worldcover_file,
        outdir='outdir', stub='test_car', savetif=False, plot=False,
    )

    assert set(ds.data_vars) == {'woody_veg', 'opportunities'}
    unique_vals = set(np.unique(ds['opportunities'].values))
    assert unique_vals.issubset({0, 5, 6, 7, 8})


def test_opportunities_contour_spacing():
    """Test that different contour spacings produce different results."""
    ds_cs5 = opportunities(
        tree_file, roads_data=roads_file, gullies_data=gullies_file,
        dem_data=dem_file, worldcover_data=worldcover_file,
        outdir='outdir', stub='test_cs5', savetif=False, plot=False, contour_spacing=5,
    )
    ds_cs20 = opportunities(
        tree_file, roads_data=roads_file, gullies_data=gullies_file,
        dem_data=dem_file, worldcover_data=worldcover_file,
        outdir='outdir', stub='test_cs20', savetif=False, plot=False, contour_spacing=20,
    )

    n_cs5 = (ds_cs5['opportunities'].values == 8).sum()
    n_cs20 = (ds_cs20['opportunities'].values == 8).sum()
    # More contours (smaller spacing) should produce at least as many contour opportunities
    assert n_cs5 >= n_cs20


def test_opportunities_equal_area():
    """Test that equal_area=True runs without error and produces valid output."""
    ds = opportunities(
        tree_file, roads_data=roads_file, gullies_data=gullies_file,
        dem_data=dem_file, worldcover_data=worldcover_file,
        outdir='outdir', stub='test_ea', savetif=False, plot=False, equal_area=True,
    )

    assert set(ds.data_vars) == {'woody_veg', 'opportunities'}
    unique_vals = set(np.unique(ds['opportunities'].values))
    assert unique_vals.issubset({0, 5, 6, 7, 8})
