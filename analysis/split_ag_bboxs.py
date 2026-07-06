"""
Filter the Australia-wide BARRA prediction tile grid down to tiles that intersect
agricultural (grazing / no-BWH) regions and split into fixed-size chunks.
"""

import geopandas as gpd

AUS_BBOXS = "/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_aus.gpkg"
AG_MASK = "/g/data/xe2/cb8590/Nick_outlines/barra_bboxs_grazing_no_bwh.gpkg"
AG_BBOXS_OUT = "/g/data/xe2/cb8590/Outlines/BARRA_bboxs/barra_bboxs_ag.gpkg"

CHUNK_DIR = "/g/data/xe2/cb8590/Outlines/BARRA_bboxs/BARRA_bboxs_ag_noxy_df_4326_2025"
CHUNK_STUB = "BARRA_bboxs_ag_noxy_df_4326_2025"
CHUNK_SIZE = 800  # The more jobs the worse performance per job because of blocking conflicts. But also want to have compute stay within the 24 hour time limit.


def main():
    print(f"Loading {AUS_BBOXS}")
    gdf_aus = gpd.read_file(AUS_BBOXS)
    print(f"  {len(gdf_aus):,} tiles")

    print(f"Loading ag mask {AG_MASK}")
    gdf_mask = gpd.read_file(AG_MASK).to_crs(gdf_aus.crs)
    print(f"  {len(gdf_mask):,} mask polygons")

    print("Spatial join (intersects)...")
    joined = gpd.sjoin(gdf_aus, gdf_mask[['geometry']], how='inner', predicate='intersects')
    gdf_ag = gdf_aus.loc[joined.index.unique()].reset_index(drop=True)
    print(f"Kept {len(gdf_ag):,} / {len(gdf_aus):,} tiles")

    gdf_ag.to_file(AG_BBOXS_OUT)
    print(f"Saved: {AG_BBOXS_OUT}")

    import os
    os.makedirs(CHUNK_DIR, exist_ok=True)
    n = len(gdf_ag)
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = gdf_ag.iloc[start:end]
        out = f"{CHUNK_DIR}/{CHUNK_STUB}_{start}-{end}.gpkg"
        chunk.to_file(out)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Wrote {n_chunks} chunk gpkgs of up to {CHUNK_SIZE} rows to {CHUNK_DIR}")


if __name__ == '__main__':
    main()
