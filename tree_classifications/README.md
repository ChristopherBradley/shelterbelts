# Preprocessing Steps:

1. Run generate_tilesx25.py to create a list of coordinates for downloading tiles
2. ./main_tasmania.sh to download sentinel imagery for each of the 125 tiles
    - This calls run_sentinel.sh which calls sentinel_download_km.py
3. Run tile_outputs.py to create output data for each of the 125 tiles
4. Run merge_inputs_outputs to create a DataFrame with all the data