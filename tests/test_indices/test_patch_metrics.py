import os

import numpy as np
import pandas as pd
import rioxarray as rxr

from shelterbelts.indices.patch_metrics import patch_metrics, linear_categories_labels


stub = 'g2_26729'


def test_patch_metrics_basic():
    """Basic test for patch_metrics function"""
    ds, df = patch_metrics(
        f"data/{stub}_gullies_and_roads_buffer_categories.tif",
        outdir="outdir",
        stub=stub
    )
    assert os.path.exists(f"outdir/{stub}_linear_categories.tif")
    assert os.path.exists(f"outdir/{stub}_linear_categories.png")
    assert os.path.exists(f"outdir/{stub}_assigned_labels.tif")
    assert os.path.exists(f"outdir/{stub}_patch_metrics.csv")


def test_patch_metrics_category_name_not_nan():
    """Verify that category_name is populated for every row in patch_metrics output."""
    _, df = patch_metrics(
        f"data/{stub}_gullies_and_roads_buffer_categories.tif",
        outdir="outdir",
        stub=stub,
        plot=False,
        save_csv=False,
        save_tif=False,
        save_labels=False,
    )
    assert len(df) > 0, "patch_metrics returned an empty DataFrame"
    assert "category_name" in df.columns, "category_name column is missing"
    assert df["category_name"].notna().all(), (
        f"category_name has NaN values:\n{df[df['category_name'].isna()]}"
    )
    # Every category_name should be a valid label
    valid_names = set(linear_categories_labels.values())
    assert df["category_name"].isin(valid_names).all(), (
        f"Unexpected category names: {set(df['category_name']) - valid_names}"
    )


def test_patch_metrics_csv_matches_returned_df():
    """The saved CSV should be relabelled to linear/non-linear, not left as the
    intermediate 'Other Trees' (category 14) placeholder.
    """
    _, df = patch_metrics(
        f"data/{stub}_gullies_and_roads_buffer_categories.tif",
        outdir="outdir",
        stub=stub,
        plot=False,
        save_csv=True,
        save_tif=False,
        save_labels=False,
    )
    csv_df = pd.read_csv(f"outdir/{stub}_patch_metrics.csv")

    assert not (csv_df["category_id"] == 14).any(), (
        "patch_metrics.csv still contains rows with the intermediate "
        "'Other Trees' (category_id 14) label:\n"
        f"{csv_df[csv_df['category_id'] == 14]}"
    )

    # The saved CSV should match what the function returns in memory
    pd.testing.assert_frame_equal(
        csv_df.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False
    )


def test_scattered_trees_match_tree_categories():
    """patch_metrics should not invent new Scattered Trees (should be the same as the original tree categories)"""
    ds, _ = patch_metrics(
        f"data/{stub}_gullies_and_roads_buffer_categories.tif",
        outdir="outdir",
        stub=stub,
        plot=False,
        save_csv=False,
        save_tif=False,
        save_labels=False,
        save_gpkg=False,
    )
    da_tree = rxr.open_rasterio(f"data/{stub}_tree_categories.tif").isel(band=0)

    scattered_tree = da_tree.values == 11
    scattered_linear = ds['linear_categories'].values == 11
    assert scattered_tree.sum() > 0
    assert np.array_equal(scattered_linear, scattered_tree)
