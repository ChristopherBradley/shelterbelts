import requests


# +
# %%time
# 1) get source by ISSN (example for JOSS)
resp = requests.get("https://api.openalex.org/sources/issn:2475-9066").json()
source_id = resp['id'].split('/')[-1]   # Sxxxxx

# 2) get top cited works for that source
url = f"https://api.openalex.org/works?filter=primary_location.source.id:{resp['id']}&sort=cited_by_count:desc&per-page=10"
top = requests.get(url).json()
for w in top['results']:
    print(w['display_name'])
    print("Citations:", w['cited_by_count'])
    print("DOI:", w.get('doi'))
    print("---")

# -


