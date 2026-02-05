"""Tests for crop_and_rasterize utility."""

import tempfile
import pytest
import rioxarray as rxr

from shelterbelts.utils.crop_and_rasterize import crop_and_rasterize
from shelterbelts.utils.filepaths import get_filename


def test_crop_and_rasterize_hydrolines():
    """Test crop_and_rasterize with hydrolines GeoPackage."""
    tif_file = get_filename('g2_26729_tree_categories.tif')
    hydrolines_gpkg = 'data/g2_26729_hydrolines_cropped.gpkg'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        gdf, ds = crop_and_rasterize(
            tif_file,
            hydrolines_gpkg,
            outdir=tmpdir,
            stub='test_hydrolines',
            layer='HydroLines',
            feature_name='gullies',
            save_gpkg=False,
            savetif=False
        )
        
        # Check outputs
        assert len(gdf) > 0, "Should have cropped some hydrolines"
        assert 'gullies' in ds.data_vars

def test_crop_and_rasterize_roads():
    """Test crop_and_rasterize with roads GeoPackage."""
    tif_file = get_filename('g2_26729_tree_categories.tif')
    roads_gpkg = 'data/g2_26729_roads_cropped.gpkg'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        gdf, ds = crop_and_rasterize(
            tif_file,
            roads_gpkg,
            outdir=tmpdir,
            stub='test_roads',
            layer='NationalRoads_2025_09',
            feature_name='roads',
            save_gpkg=False,
            savetif=False
        )
        
        # Check outputs
        assert len(gdf) > 0, "Should have cropped some roads"
        assert 'roads' in ds.data_vars


def test_crop_and_rasterize_with_dataarray():
    """Test crop_and_rasterize with DataArray input instead of file path."""
    # Load raster as DataArray
    tif_file = get_filename('g2_26729_tree_categories.tif')
    da = rxr.open_rasterio(tif_file).isel(band=0)
    
    hydrolines_gpkg = 'data/g2_26729_hydrolines_cropped.gpkg'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        gdf, ds = crop_and_rasterize(
            da,
            hydrolines_gpkg,
            outdir=tmpdir,
            stub='test_dataarray',
            layer='HydroLines',
            feature_name='gullies',
            save_gpkg=False,
            savetif=False
        )
        
        # Check outputs
        assert len(gdf) > 0, "Should have cropped some hydrolines"
        assert 'gullies' in ds.data_vars