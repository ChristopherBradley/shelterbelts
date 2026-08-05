#!/usr/bin/env python
"""
generate_gch_plots.py — GCH-derived equivalents of 4 of the generate_main_plots.py charts.

outdir/tif_value_counts_gch2 holds 3 separate export folders (not a single
'global' folder with default_/less_/more_ prefixed files like tif_value_counts_ag) -
one per shelter-detection run, each using a different method/threshold and its
own filename scheme:
    tif_value_counts_gch_ag_less_percentmethod  -> 'less' (low threshold, density/percent method)
    tif_value_counts_gch_ag_masked              -> 'default' (wind method, already masked at source)
    tif_value_counts_gch_ag_more_windmethod     -> 'more' (high threshold, wind method)
'less'/'more' each have a plain and a '_masked' value-counts export (masked
drops the 0/255 nodata rows so percentages are computed over valid pixels
only); 'default' only has the plain export since it's already masked at the
source. This uses the masked variants for 'less'/'more' to match.

Since 'less' comes from the density/percent method, it has no tree-type
breakdown (only the generic 30/32/40/42 codes), unlike 'default'/'more'
(wind method, codes 30/32-39/40-49) - so the 'less' bars/error-bar edge for
tree-type-specific categories will read as 0. That's expected given the
requested method mix, not a bug.

Usage:
    python analysis/generate_gch_plots.py
"""

from category_area_summary import (
    category_area_summary,
    plot_category_bars,
    plot_totals_panels,
    SHELTER_PALETTE,
)
from generate_main_plots import SHELTER_ORDER
from shelterbelts.indices.shelter_categories import shelter_categories_cmap

GCH_ROOT = "outdir/tif_value_counts_gch2"
LESS_DIR = f"{GCH_ROOT}/tif_value_counts_gch_ag_less_percentmethod"
DEFAULT_DIR = f"{GCH_ROOT}/tif_value_counts_gch_ag_masked"
MORE_DIR = f"{GCH_ROOT}/tif_value_counts_gch_ag_more_windmethod"

SHELTER_CATEGORIES_METHODS = {
    "less": f"{LESS_DIR}/gch_less_percentmethod_merged_shelter_categories_masked_value_counts.csv",
    "default": f"{DEFAULT_DIR}/gch_windmethod_merged_shelter_categories_value_counts.csv",
    "more": f"{MORE_DIR}/gch_more_windmethod_merged_shelter_categories_masked_value_counts.csv",
}
OPPORTUNITIES_METHODS = {
    "less": f"{LESS_DIR}/gch_less_percentmethod_merged_opportunities_masked_value_counts.csv",
    "default": f"{DEFAULT_DIR}/gch_windmethod_merged_opportunities_value_counts.csv",
    "more": f"{MORE_DIR}/gch_more_windmethod_merged_opportunities_masked_value_counts.csv",
}

TITLE_SUFFIX = " (GCH)"


def main():
    # 1. Tree categories (11-19), Australia-wide.
    out_stub = f"{GCH_ROOT}/global_11-19_percentmethod_merged_shelter_categories_hectares"
    table = category_area_summary(None, None, "11-19", methods=SHELTER_CATEGORIES_METHODS,
                                   units="hectares", order_by="default")
    table.to_csv(f"{out_stub}.csv")
    print(f"Saved: {out_stub}.csv")
    plot_category_bars(None, None, "11-19", methods=SHELTER_CATEGORIES_METHODS,
                        palette=SHELTER_PALETTE, order_by="default",
                        title=f"Tree Categories - Australia{TITLE_SUFFIX}",
                        output_png=f"{out_stub}.png")

    # 2. Grassland categories (30-39), Australia-wide. 'less' (percentmethod)
    # has no tree-type breakdown, so its tree-type bars read as 0.
    grassland_order = ["Unsheltered Grassland"] + [s.format(kind="Grassland") for s in SHELTER_ORDER]
    out_stub = f"{GCH_ROOT}/global_30-39_windmethod_merged_shelter_categories_hectares"
    table = category_area_summary(None, None, "30,33-39", methods=SHELTER_CATEGORIES_METHODS,
                                   units="hectares", order_by=None)
    table = table.reindex(grassland_order)
    table.to_csv(f"{out_stub}.csv")
    print(f"Saved: {out_stub}.csv")
    plot_category_bars(None, None, "30,33-39", methods=SHELTER_CATEGORIES_METHODS,
                        category_order=grassland_order, palette=shelter_categories_cmap,
                        title=f"Grassland Categories - Australia{TITLE_SUFFIX}",
                        output_png=f"{out_stub}.png")

    # 3. Planting opportunities vs existing shelter, near gullies and roads.
    def gullies_roads_bars(methods):
        return [
            dict(label="Trees in Gullies", color=shelter_categories_cmap[15],
                 input_dir=None, suffix=None, value_range="15", methods=methods),
            dict(label="Trees next to Roads", color=shelter_categories_cmap[17],
                 input_dir=None, suffix=None, value_range="17", methods=methods),
            dict(label="Sheltered by Trees in Gullies", color=shelter_categories_cmap[35],
                 input_dir=None, suffix=None, value_range="35,45", methods=methods),
            dict(label="Sheltered by Trees next to Roads", color=shelter_categories_cmap[37],
                 input_dir=None, suffix=None, value_range="37,47", methods=methods),
        ]

    plot_totals_panels(
        panels=[
            dict(bars=gullies_roads_bars(OPPORTUNITIES_METHODS),
                 title=f"Planting Opportunities - Australia{TITLE_SUFFIX}"),
            dict(bars=gullies_roads_bars(SHELTER_CATEGORIES_METHODS),
                 title=f"Existing Shelter - Australia{TITLE_SUFFIX}"),
        ],
        output_png=f"{GCH_ROOT}/global_opportunities_vs_shelter_gullies_roads_hectares.png",
    )

    # 4. Sheltered vs unsheltered totals for grassland and cropland.
    from category_area_summary import plot_totals_bars, CROPLAND_PALETTE
    from generate_main_plots import _darken

    grassland_sheltered_color = _darken(shelter_categories_cmap[30], 0.35)
    cropland_sheltered_color = _darken(CROPLAND_PALETTE[40], 0.35)
    plot_totals_bars(
        bars=[
            dict(label="Unsheltered Grassland", color=shelter_categories_cmap[30],
                 input_dir=None, suffix=None, value_range="30", methods=SHELTER_CATEGORIES_METHODS),
            dict(label="Unsheltered Cropland", color=CROPLAND_PALETTE[40],
                 input_dir=None, suffix=None, value_range="40", methods=SHELTER_CATEGORIES_METHODS),
            dict(label="Sheltered Grassland (total)", color=grassland_sheltered_color,
                 input_dir=None, suffix=None, value_range="32-39", methods=SHELTER_CATEGORIES_METHODS),
            dict(label="Sheltered Cropland (total)", color=cropland_sheltered_color,
                 input_dir=None, suffix=None, value_range="42-49", methods=SHELTER_CATEGORIES_METHODS),
        ],
        title=f"Sheltered vs Unsheltered Totals - Australia{TITLE_SUFFIX}",
        output_png=f"{GCH_ROOT}/global_sheltered_vs_unsheltered_totals_hectares.png",
    )


if __name__ == "__main__":
    main()
