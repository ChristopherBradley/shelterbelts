# +
# Demo-ing each of the API downloads. 
# Going to convert this to a jupyter notebook once finished.

# Each API has at least 3 functions:
# - download via bbox
# - download via lat, lon, buffer
# - quick visualisation

# Each API has been tested using singlepoint, 1km, 10km, 100km, 1000km areas. 
# Each API has been tested with valid and invalid coordinates.
# Each API gives an estimate of the download time if the area is large.

# I have deliberately included example inputs wherever possible, because I find this makes it easier for me to use and debug each function.
# I have also made the inputs for each function as consistent as possible, even if the API's each work differently under the hood. 

# -

# Change directory to this repo - this should work on gadi or locally via python or jupyter.
# Unfortunately, this needs to be in all files that can be run directly & use local imports.
import os, sys
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
    repo_dir = os.path.dirname(os.getcwd())       
else:                                             # Already running from root of this repo. 
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)

# %%time
# Should time how long each of the imports takes
from shelterbelt_identification.worldcover import worldcover_centerpoint

outdir = 'data'
stub = 'Demo'

fulham = {'lat':-42.887122, 'lon':147.760717}

# +
# %%time
da = worldcover_centerpoint(fulham['lat'], fulham['lon'], 0.2, outdir, stub)

# 5 secs for 0.05 buffer
# 5, 7 secs for 0.1 buffer
# 30 secs for 0.2 buffer

# . secs for 0.5 buffer (takes forever)
# . secs for 1 buffer

# Anything that takes over a minute is too long and should be tiled

# -

print(f"In-memory size: {da.nbytes / 1e6:.2f} MB")


# %%time
st = speedtest.Speedtest()
download_speed = st.download() / 1_000_000  # bits to megabits
print(f"Download speed: {download_speed:.2f} Mbps")
