import os

from shelterbelts.classifications.bounding_boxes import bounding_boxes
from shelterbelts.utils.filepaths import _repo_root

multi_tif_folder = str(_repo_root / 'data' / 'multiple_binary_tifs')


def test_bounding_boxes_basic():
    gdf = bounding_boxes(
        multi_tif_folder,
        outdir='outdir',
        stub='bbox_basic',
        filetype='.tiff',
        verbose=False,
    )
    assert len(gdf) == 2
    assert {'filename', 'height', 'width', 'geometry', 'bad_tif'}.issubset(gdf.columns)
    assert os.path.exists('outdir/bbox_basic_footprints.gpkg')


def test_bounding_boxes_cover_threshold():
    """Test bounding_boxes with tif_cover_threshold adds percent_trees column and flags bad tifs."""
    gdf = bounding_boxes(
        multi_tif_folder,
        outdir='outdir',
        stub='bbox_cover',
        filetype='.tiff',
        tif_cover_threshold=5,
        verbose=False,
    )
    assert 'percent_trees' in gdf.columns
    assert 'bad_tif' in gdf.columns


def test_bounding_boxes_size_threshold():
    """Test bounding_boxes with a large size_threshold flags all tiles as bad."""
    gdf = bounding_boxes(
        multi_tif_folder,
        outdir='outdir',
        stub='bbox_size',
        filetype='.tiff',
        size_threshold=300,
        verbose=False,
    )
    assert 'bad_tif' in gdf.columns
    assert gdf['bad_tif'].all()


def test_bounding_boxes_save_centroids_writes_gpkg():
    """Test bounding_boxes writes centroid GeoPackage when save_centroids=True."""
    gdf = bounding_boxes(
        multi_tif_folder,
        outdir='outdir',
        stub='bbox_centroids',
        filetype='.tiff',
        save_centroids=True,
        verbose=False,
    )
    assert os.path.exists(f'outdir/bbox_centroids_centroids.gpkg')
