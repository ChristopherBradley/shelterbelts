import os

from shelterbelts.indices.shelter_metrics import patch_metrics, class_metrics, linear_categories_labels


stub = 'g2_26729'


def test_patch_metrics_basic():
    """Basic test for patch_metrics function"""
    ds, df = patch_metrics(
        f"data/{stub}_buffer_categories.tif",
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
        f"data/{stub}_buffer_categories.tif",
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


def test_class_metrics_basic():
    """Basic test for class_metrics function"""
    dfs = class_metrics(
        f"data/{stub}_linear_categories.tif",
        outdir="outdir",
        stub=stub,
        save_excel=True
    )
    assert os.path.exists(f"outdir/{stub}_class_metrics.xlsx")
