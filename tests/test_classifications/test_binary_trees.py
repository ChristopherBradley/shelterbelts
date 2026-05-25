import os

import rioxarray as rxr

from shelterbelts.classifications.binary_trees import canopy_height_trees, worldcover_trees

worldcover_tif = 'data/g2_26729_worldcover.tif'
chm_tif = 'data/milgadara_1kmx1km_CHM_1m.tif'


def test_worldcover_trees_basic():
    ds = worldcover_trees(worldcover_tif, outdir='outdir', stub='wc_basic', savetif=True, plot=False)
    assert 'woody_veg' in ds.data_vars
    assert ds['woody_veg'].dtype.name == 'uint8'
    assert os.path.exists('outdir/wc_basic_woody_veg.tif')
    assert set(ds['woody_veg'].values.flatten().tolist()).issubset({0, 1})

def test_worldcover_trees_no_savetif():
    ds = worldcover_trees(worldcover_tif, outdir='outdir', stub='wc_nosave', savetif=False, plot=False)
    assert not os.path.exists('outdir/wc_nosave_woody_veg.tif')

def test_worldcover_trees_from_dataarray():
    da = rxr.open_rasterio(worldcover_tif).isel(band=0).drop_vars('band')
    ds = worldcover_trees(da, stub='wc_da', savetif=False, plot=False)
    assert 'woody_veg' in ds.data_vars


def test_canopy_height_trees_basic():
    ds = canopy_height_trees(chm_tif, outdir='outdir', stub='chm_basic', savetif=True, plot=False)
    assert 'woody_veg' in ds.data_vars
    assert ds['woody_veg'].dtype.name == 'uint8'
    assert os.path.exists('outdir/chm_basic_woody_veg.tif')
    assert set(ds['woody_veg'].values.flatten().tolist()).issubset({0, 1})

def test_canopy_height_trees_no_savetif():
    ds = canopy_height_trees(chm_tif, outdir='outdir', stub='chm_nosave', savetif=False, plot=False)
    assert not os.path.exists('outdir/chm_nosave_woody_veg.tif')

def test_canopy_height_trees_from_dataarray():
    da = rxr.open_rasterio(chm_tif).isel(band=0).drop_vars('band')
    ds = canopy_height_trees(da, stub='chm_da', savetif=False, plot=False)
    assert 'woody_veg' in ds.data_vars