#!/usr/bin/env python
"""
generate_main_plots.py — regenerate the main category-area plots.

A record of the one-off python -c calls used to build up these charts
interactively, so they can be reproduced with a single command:
    python analysis/generate_main_plots.py

For each OUTDIR in OUTDIRS, produces (alongside its global/states subfolders):
    global_11-19_percentmethod_merged_shelter_categories_hectares.{csv,png}
    states_grid_11-19_percentmethod_merged_shelter_categories_hectares.png
    global_30-39_windmethod_merged_shelter_categories_hectares.csv
    global_40-49_windmethod_merged_shelter_categories_hectares.csv
    global_sheltered_by_tree_type_hectares.png       (grassland + cropland, sheltered only)
    global_sheltered_vs_unsheltered_totals_hectares.png
"""

from category_area_summary import (
    category_area_summary,
    plot_category_bars,
    plot_category_grid,
    plot_category_panels,
    plot_totals_bars,
    SHELTER_PALETTE,
    CROPLAND_PALETTE,
)
from shelterbelts.indices.shelter_categories import shelter_categories_cmap

# Each entry: (folder, title suffix). tif_value_counts_ag holds the same data
# masked to 250m cropping/grazing NLUM pixels only.
OUTDIRS = [
    ("outdir/tif_value_counts", ""),
    ("outdir/tif_value_counts_ag", ""),
]


def _darken(rgb, amount):
    """Darken an (r, g, b) colour towards black by `amount` (0-1)."""
    return tuple(round(c * (1 - amount)) for c in rgb)

# Fixed row order for the grassland/cropland charts, matching the tree
# categories chart's default-sorted order (Non-linear, Patch Edge, Gullies,
# Roads, Linear), with the unsheltered bucket first.
SHELTER_ORDER = [
    "{kind} sheltered by Non-linear Patches",
    "{kind} sheltered by Trees in Gullies",
    "{kind} sheltered by Patch Edge",
    "{kind} sheltered by Trees next to Roads",
    "{kind} sheltered by Linear Patches",
]


def save_table(input_dir, suffix, value_range, out_stub, order_by, category_order):
    table = category_area_summary(input_dir, suffix, value_range, units="hectares",
                                   order_by=order_by)
    if category_order is not None:
        table = table.reindex(category_order)
    table.to_csv(f"{out_stub}.csv")
    print(f"Saved: {out_stub}.csv")


def save_table_and_plot(input_dir, suffix, value_range, out_stub, order_by,
                         category_order, palette, title):
    save_table(input_dir, suffix, value_range, out_stub, order_by, category_order)
    plot_category_bars(input_dir, suffix, value_range, palette=palette,
                        order_by=order_by, category_order=category_order,
                        title=title, output_png=f"{out_stub}.png")


def generate_plots(outdir, title_suffix=""):
    # 1. Tree categories (11-19), Australia-wide, percentmethod only.
    save_table_and_plot(
        input_dir=f"{outdir}/global",
        suffix="percentmethod_merged_shelter_categories.csv",
        value_range="11-19",
        out_stub=f"{outdir}/global_11-19_percentmethod_merged_shelter_categories_hectares",
        order_by="default", category_order=None,
        palette=SHELTER_PALETTE,
        title=f"Tree Categories - Australia{title_suffix}",
    )

    # 2. Tree categories (11-19), one panel per state, same 4x2 layout.
    layout = [
        ["Western Australia", "Northern Territory", "Queensland", "New South Wales"],
        ["South Australia", "Victoria", "Tasmania", "Australian Capital Territory"],
    ]
    plot_category_grid(
        input_dir=f"{outdir}/states",
        suffix="percentmethod_merged_shelter_categories.csv",
        value_range="11-19",
        layout=layout,
        palette=SHELTER_PALETTE,
        suptitle=f"Tree Categories by State{title_suffix}",
        output_png=f"{outdir}/states_grid_11-19_percentmethod_merged_shelter_categories_hectares.png",
    )

    # 3. Grassland categories (30-39) and cropland categories (40-49),
    # Australia-wide, windmethod (tree-type breakdown). Value 32/42
    # ('Sheltered Grassland'/'Cropland') are known-buggy legacy density-method
    # codes that shouldn't have any pixels here, so the full-detail CSVs skip
    # them (value_range goes straight from 30/40 to 33/43).
    grassland_order = ["Unsheltered Grassland"] + [s.format(kind="Grassland") for s in SHELTER_ORDER]
    cropland_order = ["Unsheltered Cropland"] + [s.format(kind="Cropland") for s in SHELTER_ORDER]
    windmethod_suffix = "windmethod_merged_shelter_categories.csv"
    save_table(f"{outdir}/global", windmethod_suffix, "30,33-39",
              f"{outdir}/global_30-39_windmethod_merged_shelter_categories_hectares",
              order_by=None, category_order=grassland_order)
    save_table(f"{outdir}/global", windmethod_suffix, "40,43-49",
              f"{outdir}/global_40-49_windmethod_merged_shelter_categories_hectares",
              order_by=None, category_order=cropland_order)

    # 4. Sheltered-only breakdown (unsheltered bucket dropped) for grassland
    # and cropland, stacked in one figure so they're easy to compare directly.
    sheltered_grassland_order = [s.format(kind="Grassland") for s in SHELTER_ORDER]
    sheltered_cropland_order = [s.format(kind="Cropland") for s in SHELTER_ORDER]
    plot_category_panels(
        panels=[
            dict(input_dir=f"{outdir}/global", suffix=windmethod_suffix, value_range="33-39",
                 category_order=sheltered_grassland_order, palette=shelter_categories_cmap,
                 title=f"Sheltered Grassland by Tree Type - Australia{title_suffix}"),
            dict(input_dir=f"{outdir}/global", suffix=windmethod_suffix, value_range="43-49",
                 category_order=sheltered_cropland_order, palette=CROPLAND_PALETTE,
                 title=f"Sheltered Cropland by Tree Type - Australia{title_suffix}"),
        ],
        output_png=f"{outdir}/global_sheltered_by_tree_type_hectares.png",
    )

    # 5. Complementary totals chart: unsheltered vs. total sheltered (summed
    # across all tree types, incl. the legacy 32/42 bucket) for grassland and
    # cropland side by side, all on one shared y-scale. The 'sheltered' bars
    # use a darkened version of the unsheltered base colour (rather than the
    # per-tree-type blend, which goes muddy grey for cropland - pink and
    # near-black tree greens are far apart in hue) so each pair reads as a
    # light/dark shade of the same grassland/cropland family colour.
    grassland_sheltered_color = _darken(shelter_categories_cmap[30], 0.35)
    cropland_sheltered_color = _darken(CROPLAND_PALETTE[40], 0.35)
    plot_totals_bars(
        bars=[
            dict(label="Unsheltered Grassland", color=shelter_categories_cmap[30],
                 input_dir=f"{outdir}/global", suffix=windmethod_suffix, value_range="30"),
            dict(label="Unsheltered Cropland", color=CROPLAND_PALETTE[40],
                 input_dir=f"{outdir}/global", suffix=windmethod_suffix, value_range="40"),
            dict(label="Sheltered Grassland (total)", color=grassland_sheltered_color,
                 input_dir=f"{outdir}/global", suffix=windmethod_suffix, value_range="32-39"),
            dict(label="Sheltered Cropland (total)", color=cropland_sheltered_color,
                 input_dir=f"{outdir}/global", suffix=windmethod_suffix, value_range="42-49"),
        ],
        title=f"Sheltered vs Unsheltered Totals - Australia{title_suffix}",
        output_png=f"{outdir}/global_sheltered_vs_unsheltered_totals_hectares.png",
    )


def main():
    for outdir, title_suffix in OUTDIRS:
        generate_plots(outdir, title_suffix)


if __name__ == "__main__":
    main()
