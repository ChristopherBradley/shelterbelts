import os

from shelterbelts.indices.class_metrics import class_metrics


stub = 'g2_26729'


def test_class_metrics_basic():
    """Basic test for class_metrics function"""
    dfs = class_metrics(
        f"data/{stub}_linear_categories.tif",
        outdir="outdir",
        stub=stub,
        save_excel=True
    )
    assert os.path.exists(f"outdir/{stub}_class_metrics.xlsx")
