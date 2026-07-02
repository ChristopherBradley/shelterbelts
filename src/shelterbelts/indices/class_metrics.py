import os
import argparse

import pandas as pd
import rioxarray as rxr
import xarray as xr

from shelterbelts.indices.shelter_categories import shelter_categories_labels


# Mapping for broader categories
landcover_groups = {
    10: "Trees",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    80: "Water"
}
def group_label(cat_id):
    """Apply broader group categories to each pixel"""
    return landcover_groups.get((cat_id // 10) * 10, 'Other')  # Flooring to the nearest 10


# Need to make sure you've pip installed xlsxwriter for this to work
def save_excel_sheets(dfs, filename):
    """Save each dataframe in a separate sheet in an excel file, and resize columns nicely
    dfs should be a dictionary of dataframes."""
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:

        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]

            # Set the column widths to the longest item in that column
            for i, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                )
                worksheet.set_column(i + 1, i + 1, max_len)

            # Also set the index column width
            max_idx_len = max(df.index.astype(str).map(len).max(), len(str(df.index.name or "")))
            worksheet.set_column(0, 0, max_idx_len)

    print("Saved:", filename)


def class_metrics(shelter_data, outdir=".", stub="TEST", save_excel=True):
    """Calculate the percentage cover in each class.

    Parameters
    ----------
        shelter_data : str, xarray.Dataset, or xarray.DataArray
            A tif file, Dataset, or DataArray where the integers represent the categories defined in
            'shelter_categories_labels' (the output of shelter_categories.py). Both the current (30/40
            plus 32-39/42-49) and legacy (31/32/41/42) farmland encodings are handled.
        outdir : str, optional
            The output directory to save results.
        stub : str, optional
            Prefix for output file names.
        save_excel : bool, optional
            Whether to save the results as an Excel file with multiple sheets.

    Returns
    -------
    dict
        A dictionary with the following pandas DataFrames:

        - **Overall**: Count and percentage for each category
        - **Landcover**: Aggregated statistics by landcover group (Trees, Grassland, Cropland, Built-up, Water)
        - **Trees**: Breakdown of tree categories as percentage of total tree coverage
        - **Shelter**: Percentage of grassland and cropland that are sheltered vs unsheltered

    Examples
    --------
    Using file paths as input:

    >>> from shelterbelts.utils.filepaths import get_filename
    >>> linear_file = get_filename('g2_26729_linear_categories.tif')
    >>> dfs = class_metrics(linear_file, outdir='/tmp', save_excel=False)
    >>> len(dfs) == 4
    True

    Using a Dataset carrying the shelter_categories band:

    >>> import rioxarray as rxr
    >>> da = rxr.open_rasterio(linear_file).squeeze('band').drop_vars('band')
    >>> ds = da.to_dataset(name='shelter_categories')
    >>> dfs = class_metrics(ds, outdir='/tmp', save_excel=False)
    >>> len(dfs) == 4
    True

    Downloads
    ---------
        class_metrics.xlsx: An excel file with each dataframe in a separate tab.

    """
    if isinstance(shelter_data, str):
        da = rxr.open_rasterio(shelter_data).isel(band=0)
    elif isinstance(shelter_data, xr.Dataset):
        band = 'shelter_categories' if 'shelter_categories' in shelter_data.data_vars else 'linear_categories'
        da = shelter_data[band]
    else:
        da = shelter_data

    # Overall stats per category
    counts = da.values.ravel()
    counts = pd.Series(counts).value_counts().sort_index()
    total_pixels = da.shape[0] * da.shape[1]
    df_overall = pd.DataFrame({
        'category_id': counts.index,
        'label': [shelter_categories_labels.get(cat, 'Unknown') for cat in counts.index],
        'pixel_count': counts.values,
        'percentage': (counts.values / total_pixels) * 100
    })
    df_overall = df_overall.set_index('label')
    df_overall['percentage'] = df_overall['percentage'].round(2)
    df_overall = df_overall.sort_values(by='percentage', ascending=False)

    # Landcover groups
    df_overall['landcover_group'] = df_overall['category_id'].apply(group_label) # Not sure why this was commented out?
    df_landcover = df_overall.groupby('landcover_group')[['pixel_count', 'percentage']].sum()
    df_landcover = df_landcover.sort_values(by='percentage', ascending=False)
    df_landcover['percentage'] = df_landcover['percentage'].round(2)

    # Tree groups
    df_trees = df_overall[df_overall['landcover_group'] == 'Trees'].copy()
    total_tree_pixels = df_trees['pixel_count'].sum()
    df_trees['percentage'] = (df_trees['pixel_count'] / total_tree_pixels) * 100
    df_trees['percentage'] = df_trees['percentage'].round(2)
    df_trees = df_trees.drop(columns=['category_id', 'landcover_group'])
    df_trees = df_trees.sort_values(by='percentage', ascending=False)

    # Shelter groups, derived from the category codes so it is robust to the exact labels:
    #   grassland unsheltered=30 (legacy 31), sheltered=32-39;  cropland unsheltered=40 (legacy 41), sheltered=42-49
    code_counts = pd.Series(da.values.ravel()).value_counts()
    def total(codes):
        return int(sum(int(code_counts.get(c, 0)) for c in codes))
    df_summary = pd.DataFrame({
        'Sheltered':   {'Grassland': total(range(32, 40)), 'Cropland': total(range(42, 50))},
        'Unsheltered': {'Grassland': total([30, 31]),      'Cropland': total([40, 41])},
    })
    df_summary.loc['Total'] = df_summary.sum()
    df_shelter = df_summary.div(df_summary.sum(axis=1).replace(0, pd.NA), axis=0) * 100
    df_shelter = df_shelter.round(2)

    dfs = {
        "Overall": df_overall,
        "Landcover": df_landcover,
        "Trees": df_trees,
        "Shelter": df_shelter,
    }

    if save_excel:
        filename = os.path.join(outdir, f"{stub}_class_metrics.xlsx")
        save_excel_sheets(dfs, filename)

    return dfs


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()

    parser.add_argument('shelter_data', help='Integer tif file generated by shelter_categories.py')
    parser.add_argument('--outdir', default='.', help='Output directory for saving results (default: current directory)')
    parser.add_argument('--stub', default='TEST', help='Prefix for output filenames (default: TEST)')
    parser.add_argument('--no-save-excel', dest='save_excel', action='store_false', default=True, help='Disable Excel output (default: enabled)')

    return parser


if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    dfs = class_metrics(args.shelter_data, args.outdir, args.stub, save_excel=args.save_excel)
