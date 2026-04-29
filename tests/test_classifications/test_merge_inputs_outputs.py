import os

from shelterbelts.classifications.merge_inputs_outputs import merge_inputs_outputs, jittered_grid, aggregated_metrics
from shelterbelts.utils.filepaths import get_filename

import pickle
import rioxarray as rxr

sentinel_pickle = get_filename('g2_019_sentinel_150mx150m.pkl')
tree_tif        = get_filename('g2_019_binary_trees_150mx150m.tiff')


def test_merge_inputs_outputs_columns():
    """Output CSV has the expected feature columns and binary tree label."""
    df = merge_inputs_outputs(sentinel_pickle, tree_tif, outdir='outdir', spacing=3)
    assert 'tree_cover' in df.columns
    assert 'NDVI_temporal_median' in df.columns
    assert 'nbart_red_focal_mean' in df.columns
    assert len(df) > 0
    assert os.path.exists('outdir/g2_019_binary_trees_150mx150m_df_r4_s3_2020.csv')


def test_merge_inputs_outputs_dtypes():
    """Feature columns are float32 and tree_cover is numeric."""
    df = merge_inputs_outputs(sentinel_pickle, tree_tif, outdir='outdir', spacing=3)
    assert df['NDVI_temporal_median'].dtype.name == 'float32'
    assert set(df['tree_cover'].dropna().unique()).issubset({0.0, 1.0})


def test_jittered_grid_spacing():
    """A smaller spacing produces more sample rows than a larger spacing."""
    with open(sentinel_pickle, 'rb') as f:
        ds = pickle.load(f)
    da = rxr.open_rasterio(tree_tif).isel(band=0).drop_vars('band')
    ds = ds.rio.reproject_match(da)
    ds['tree_cover'] = da.astype(float)
    ds = aggregated_metrics(ds)

    df_fine   = jittered_grid(ds, spacing=2)
    df_coarse = jittered_grid(ds, spacing=5)
    assert len(df_fine) > len(df_coarse)
