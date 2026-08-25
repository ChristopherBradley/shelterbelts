"""Build a footprints GeoPackage for the Global_Canopy_Height_v2 tiles (10-digit ids).

The bundled global_canopy_height_footprints.gpkg is for a different (9-digit) tileset, so we index
the v2 tiles' bounds ourselves (metadata-only reads, fast). Output columns: filename, geometry (EPSG:3857).

    python gch_v2_footprints.py
"""
import os
import glob
import argparse

import rasterio
import geopandas as gpd
from shapely.geometry import box


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gch_dir', default='/scratch/xe2/cb8590/Global_Canopy_Height_v2')
    p.add_argument('--out', default='/g/data/xe2/cb8590/Outlines/global_canopy_height_v2_footprints.gpkg')
    args = p.parse_args()

    tifs = sorted(glob.glob(os.path.join(args.gch_dir, '*.tif')))
    print(f"Indexing {len(tifs)} tiles ...", flush=True)
    names, geoms, crs = [], [], None
    for i, t in enumerate(tifs):
        with rasterio.open(t) as src:
            if crs is None:
                crs = src.crs
            names.append(os.path.basename(t))
            geoms.append(box(*src.bounds))
        if i % 500 == 0:
            print(f"  {i}/{len(tifs)}", flush=True)
    gdf = gpd.GeoDataFrame({'filename': names}, geometry=geoms, crs=crs)
    gdf.to_file(args.out, driver='GPKG')
    print(f"Saved {len(gdf)} footprints ({crs}) -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
