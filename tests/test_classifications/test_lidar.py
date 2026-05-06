import os

import pytest

pdal = pytest.importorskip('pdal')

from shelterbelts.classifications.lidar import lidar
from shelterbelts.utils.filepaths import laz_sample, dem_h_sample


def test_lidar_category5_binary():
    """Binary raster using existing 'Tall Vegetation' (category 5) classifications."""
    counts, da_tree = lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_cat5',
        category5=True,
        binary=True,
        resolution=1,
    )
    assert counts is not None and da_tree is not None
    assert da_tree.dtype.name == 'uint8'
    assert set(da_tree.values.flatten().tolist()).issubset({0, 1})
    assert os.path.exists('outdir/lidar_cat5_woodyveg_res1_cat5.tif')


def test_lidar_chm_percent_cover():
    """Canopy height model and percent cover raster using pdal's hag_nn algorithm."""
    chm, da_tree = lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_CHM',
        resolution=1,
    )
    assert chm is not None and da_tree is not None
    assert da_tree.dtype.name == 'uint8'
    assert os.path.exists('outdir/lidar_CHM_chm_res1.tif')
    assert os.path.exists('outdir/lidar_CHM_percentcover_res1_height2m.tif')


def test_lidar_height_threshold():
    """Changing the height threshold filters out shorter vegetation."""
    chm, da_tree_low = lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_ht2',
        resolution=1,
        height_threshold=2,
    )
    chm, da_tree_high = lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_ht10',
        resolution=1,
        height_threshold=10,
    )
    assert int(da_tree_high.sum()) <= int(da_tree_low.sum())


def test_lidar_dem():
    """Providing a DEM skips filters.smrf and still produces a CHM and percent cover raster."""
    chm, da_tree = lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_dem',
        resolution=1,
        dem=dem_h_sample,
    )
    assert chm is not None and da_tree is not None
    assert da_tree.dtype.name == 'uint8'
    assert os.path.exists('outdir/lidar_dem_chm_res1.tif')
    assert os.path.exists('outdir/lidar_dem_percentcover_res1_height2m.tif')


def test_lidar_delineate_crowns():
    """delineate_crowns=True produces a GeoPackage of tree crown polygons."""
    import geopandas as gpd
    lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_crowns',
        resolution=1,
        delineate_crowns=True,
    )
    gpkg_path = 'outdir/lidar_crowns_crowns.gpkg'
    assert os.path.exists(gpkg_path)
    crowns = gpd.read_file(gpkg_path)
    assert len(crowns) > 0
    assert 'treeID' in crowns.columns
    assert 'mean_height_m' in crowns.columns
    assert crowns['mean_height_m'].min() > 0


def test_lidar_resolution():
    """Changing the output resolution produces a coarser percent cover raster."""
    chm, da_tree_1m = lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_res1',
        resolution=1,
    )
    chm, da_tree_5m = lidar(
        laz_sample,
        outdir='outdir',
        stub='lidar_res5',
        resolution=5,
    )
    assert da_tree_5m.dtype.name == 'uint8'
    assert os.path.exists('outdir/lidar_res5_chm_res1.tif')
    assert os.path.exists('outdir/lidar_res5_percentcover_res5_height2m.tif')
    assert da_tree_5m.shape[0] < da_tree_1m.shape[0]
    assert da_tree_5m.shape[1] < da_tree_1m.shape[1]
