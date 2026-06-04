import os
import argparse

import geopandas as gpd


def find_duplicates(gpkg, limit=None):
    """Return a dict mapping basename -> list of full filepaths, for basenames that appear more than once."""
    gdf = gpd.read_file(gpkg)
    if limit is not None:
        gdf = gdf.iloc[:limit]

    total = len(gdf)

    seen = {}
    for filepath in gdf['filepath']:
        name = os.path.basename(filepath)
        seen.setdefault(name, []).append(filepath)

    duplicates = {name: paths for name, paths in seen.items() if len(paths) > 1}
    n_with_duplicates = sum(len(paths) for paths in duplicates.values())

    print(f"Total LAZ files in GeoPackage: {total}")
    print(f"LAZ files with a duplicate basename: {n_with_duplicates} across {len(duplicates)} unique name(s)")

    return duplicates


def find_extra_folder_duplicates(gpkg, extra_folder, limit=None):
    """Return list of filepaths from the gpkg whose basename already exists in extra_folder."""
    gdf = gpd.read_file(gpkg)
    if limit is not None:
        gdf = gdf.iloc[:limit]

    total = len(gdf)

    # Build set of basenames present in extra_folder (recursive)
    extra_names = set()
    for dirpath, _, filenames in os.walk(extra_folder):
        for fname in filenames:
            extra_names.add(fname)

    print(f"Total LAZ files in GeoPackage: {total}")
    print(f"Unique filenames found in extra folder: {len(extra_names)}")

    to_remove = []
    for filepath in gdf['filepath']:
        if os.path.basename(filepath) in extra_names:
            to_remove.append(filepath)

    print(f"LAZ files in GeoPackage also present in extra folder: {len(to_remove)}")

    return to_remove


def cleanup(gpkg, remove=False, limit=None, extra_folder=None):
    if extra_folder is not None:
        to_remove = find_extra_folder_duplicates(gpkg, extra_folder, limit)
    else:
        duplicates = find_duplicates(gpkg, limit)
        if not duplicates:
            print("No duplicates found — nothing to do.")
            return
        to_remove = []
        for name, paths in duplicates.items():
            extras = paths[1:]
            for p in extras:
                to_remove.append(p)

    if not to_remove:
        print("No files to remove — nothing to do.")
        return

    print(f"\n{len(to_remove)} file(s) to remove.")

    if not remove:
        print("Dry run — pass --remove to actually delete files.")
        # for p in to_remove:
        #     print(f"  Would remove: {p}")
        return

    for p in to_remove:
        if os.path.exists(p):
            os.remove(p)
            print(f"Deleted: {p}")
        else:
            print(f"Already gone: {p}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find and remove duplicate LAZ files (same basename) listed in a GeoPackage."
    )
    parser.add_argument('gpkg', help="Path to the GeoPackage produced by laz_inventory.py")
    parser.add_argument('--remove', action='store_true',
                        help="Actually delete the duplicate files (default: dry run)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only examine the first N rows of the GeoPackage")
    parser.add_argument('--extra_folder', default=None,
                        help="Check this folder for duplicates — files found here are removed from the main gpkg paths, never from this folder")
    args = parser.parse_args()

    cleanup(args.gpkg, remove=args.remove, limit=args.limit, extra_folder=args.extra_folder)
