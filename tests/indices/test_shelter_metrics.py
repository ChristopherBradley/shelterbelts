import os
import pytest

from shelterbelts.indices.shelter_metrics import patch_metrics, class_metrics


stub = 'g2_26729'


def test_patch_metrics_basic():
    """Basic test for patch_metrics function"""
    ds, df = patch_metrics(
        f"outdir/{stub}_buffer_categories.tif",
        outdir="outdir",
        stub=stub
    )
    assert os.path.exists(f"outdir/{stub}_linear_categories.tif")
    assert os.path.exists(f"outdir/{stub}_linear_categories.png")
    assert os.path.exists(f"outdir/{stub}_labelled_categories.tif")
    assert os.path.exists(f"outdir/{stub}_labelled_categories.png")
    assert os.path.exists(f"outdir/{stub}_patch_metrics.csv")


def test_class_metrics_basic():
    """Basic test for class_metrics function"""
    dfs = class_metrics(
        f"outdir/{stub}_linear_categories.tif",
        outdir="outdir",
        stub=stub,
        save_excel=True
    )
    assert os.path.exists(f"outdir/{stub}_class_metrics.xlsx")
