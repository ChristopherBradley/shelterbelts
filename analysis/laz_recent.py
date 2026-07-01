import os
import argparse

import geopandas as gpd


def laz_recent(gpkg_path, limit=None):
    gdf = gpd.read_file(gpkg_path)
    if limit is not None:
        gdf = gdf.iloc[:limit]
    gdf = gdf.reset_index(drop=True)

    n_original = len(gdf)
    print(f"Total tiles: {n_original}")

    gdf['_basename'] = gdf['filepath'].apply(os.path.basename)

    sindex = gdf.sindex
    to_remove = set()

    for i in range(n_original):
        if i in to_remove:
            continue
        geom = gdf.geometry.iloc[i]
        name = gdf['_basename'].iloc[i]
        for j in sindex.intersection(geom.bounds):
            if j == i or j in to_remove:
                continue
            if gdf['_basename'].iloc[j] > name:
                other_geom = gdf.geometry.iloc[j]
                intersection_area = geom.intersection(other_geom).area
                overlap_fraction = intersection_area / geom.area
                if overlap_fraction > 0.9:
                    to_remove.add(i)
                    break

    n_removed = len(to_remove)
    n_kept = n_original - n_removed
    print(f"Kept:    {n_kept}")
    print(f"Removed: {n_removed}")

    gdf_recent = gdf.drop(index=list(to_remove)).drop(columns=['_basename'])

    stem, ext = os.path.splitext(gpkg_path)
    out_path = stem + '_recent' + ext
    gdf_recent.to_file(out_path, driver='GPKG')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter a LAZ tile GeoPackage to keep only the most recent tile where tiles overlap."
    )
    parser.add_argument('gpkg', help="Path to the input GeoPackage")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only examine the first N rows")
    args = parser.parse_args()

    laz_recent(args.gpkg, limit=args.limit)
