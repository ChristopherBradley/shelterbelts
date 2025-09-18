from shapely import wkt
import json
import os


def elvis_geojson(polygon, outdir='.', stub='TEST'):
    """Convert the Polygon in the email back into a geojson

    Parameters
    ----------
        polygon: The polygon string contained in the email from elevation@ga.gov.au
        outdir: Directory to save the geojson
        stub: Recommend using the same name as the zip download, e.g. DATA_587060

    Downloads
    ---------
        outdir/stub.geojson: Should be exactly the same as the geojson used in the request on elvis
    
    """
    polygon = wkt.loads(polygon)
    coords = [list(polygon.exterior.coords)]  # Shapely gives tuples; wrap in list for GeoJSON
    geojson_dict = {
        "type": "FeatureCollection",
        "name": "r1_c2",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::7844" } },
        "features": [
            {
                "type": "Feature",
                "properties": {"fid": 348},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [coords]  
                }
            }
        ]
    }
    geojson_str = json.dumps(geojson_dict, indent=2)
    output_path = os.path.join(outdir, f"{stub}.geojson")
    with open(output_path, "w") as f:
        json.dump(geojson_dict, f, indent=2)
    print("Saved:", output_path)
    return geojson_str


# +
import argparse

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--polygon', required=True, help='Polygon string from the email from elevation@ga.gov.au')
    parser.add_argument('--outdir', default='.', help='Directory to save the GeoJSON file (default: current directory)')
    parser.add_argument('--stub', default='TEST', help='Prefix for the GeoJSON filename (default: TEST)')

    return parser.parse_args()


# # %%time
if __name__ == '__main__':
    args = parse_arguments()
    
    elvis_geojson(
        polygon=args.polygon,
        outdir=args.outdir,
        stub=args.stub
    )


# +
# wkt_str = "POLYGON((149 -34.499999999999986,149 -34,148.5 -34,148.5 -34.499999999999986,149 -34.499999999999986))"
# outdir='/scratch/xe2/cb8590/tmp'
# stub = 'DATA_587060'
# # elvis_geojson(wkt_str, '/scratch/xe2/cb8590/tmp', 'DATA_587060')
# -


