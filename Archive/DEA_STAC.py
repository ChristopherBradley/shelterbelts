# +
# Following these instructions: https://knowledge.dea.ga.gov.au/guides/setup/gis/stac/
# -

import pystac_client
import odc.stac

catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")

odc.stac.configure_rio(
    cloud_defaults=True,
    aws={"aws_unsigned": True},
)

# +
# Set a bounding box
# [xmin, ymin, xmax, ymax] in latitude and longitude
bbox = [149.05, -35.32, 149.17, -35.25]

# Set a start and end date
start_date = "2021-12-01"
end_date = "2021-12-31"

# Set product ID as the STAC "collection"
collections = ["ga_ls8c_ard_3"]

# +
# Build a query with the parameters above
query = catalog.search(
    bbox=bbox,
    collections=collections,
    datetime=f"{start_date}/{end_date}",
)

# Search the STAC catalog for all items matching the query
items = list(query.items())
print(f"Found: {len(items):d} datasets")

# +
# Set up a filter query
filter_query = "eo:cloud_cover < 10"

# Query with filtering
query = catalog.search(
    bbox=bbox,
    collections=collections,
    datetime=f"{start_date}/{end_date}",
    filter=filter_query,
)

# Load our filtered data
ds_filtered = odc.stac.load(
    query.items(),
    bands=["nbart_red"],
    crs="utm",
    resolution=10,
    groupby="solar_day",
    bbox=bbox,
)

# Plot our filtered data
ds_filtered.nbart_red.plot(col="time", robust=True);
