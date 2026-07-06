import os

from shelterbelts.indices.class_metrics import class_metrics


stub = 'g2_26729'
shelter_file = f"data/{stub}_shelter_categories.tif"


def test_class_metrics_basic():
    """Basic test for class_metrics function"""
    dfs = class_metrics(
        shelter_file,
        outdir="outdir",
        stub=stub,
        save_excel=True
    )
    assert os.path.exists(f"outdir/{stub}_class_metrics.xlsx")


def test_class_metrics_shelter_split():
    """The Shelter sheet reflects the sheltered/unsheltered split from the shelter_categories band."""
    dfs = class_metrics(shelter_file, outdir="outdir", stub=stub, save_excel=False)
    df_shelter = dfs['Shelter']
    assert {'Sheltered', 'Unsheltered'} <= set(df_shelter.columns)
    # The fixture has genuinely sheltered farmland, so the split is not degenerate
    assert df_shelter.loc['Grassland', 'Sheltered'] > 0
    assert df_shelter.loc['Total', 'Sheltered'] > 0
