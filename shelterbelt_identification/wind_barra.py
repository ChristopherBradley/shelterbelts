# +
# Download wind data from BARRA for a specific region, and create a wind rose based on data from 2017-2024

# Thredds page is here: https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f2551_3726_7908_8861
# Publication is here: http://www.bom.gov.au/research/publications/researchreports/BRR-097.pdf

# uas stands for Eastward Near-Surface Wind, and vas stands for Northward Near-Surface Wind
# -

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

barra_abbreviations = {
    "uas": "Eastward Near-Surface Wind",
    "vas": "Northward Near-Surface Wind"
}


# +
def barra_singlemonth_thredds(var="uas", latitude=-34.3890427, longitude=148.469499, buffer=0.01, year="2020", month="01"):
    
    url = f"https://thredds.nci.org.au/thredds/dodsC/ob53/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{year}{month}-{year}{month}.nc"

    try:
        ds = xr.open_dataset(url, engine="netcdf4")
    except Exception as e:
        # Likely no data for the specified month
        return None

    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))

    # If the region is too small, then just find a single point
    if ds_region[var].shape[1] == 0:
        ds_region = ds.sel(lat=latitude, lon=longitude, method="nearest")
        
    return ds_region

# barra_singlemonth_thredds()


# +
# %%time
def barra_single_year(var="uas", latitude=-34.3890427, longitude=148.469499, buffer=0.01, year="2020"):
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    dss = []
    for month in months:
        ds_month = barra_singlemonth_thredds(var, latitude, longitude, buffer, year, month)
        if ds_month:
            dss.append(ds_month)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat
    
# barra_single_year()
# Took 5 secs


# +
def barra_multiyear(var="uas", latitude=-34.3890427, longitude=148.469499, buffer=0.01, years=["2020", "2021"]):
    dss = []
    for year in years:
        ds_year = barra_single_year(var, latitude, longitude, buffer, year)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat

# barra_multiyear()


# -

def barra_daily(variables=["uas", "vas"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir=".", stub="Test"):
    """Download 8day variables from BARRA at 4.4km resolution for the region/time of interest

    Parameters
    ----------
        variables: uas = Eastward Near Surface Wind, vas = Northward Near Surface Wind. See links at the top of this file for more details.
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory that the final .NetCDF gets saved.
        stub: The name to be prepended to each file download.
    
    Returns
    -------
        ds_concat: an xarray containing the requested variables in the region of interest for the time period specified
        A NetCDF file of this xarray gets downloaded to outdir/(stub)_barra_daily.nc'
    """
    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = barra_multiyear(variable, lat, lon, buffer, years)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)
    
    filename = os.path.join(outdir, f'{stub}_barra_daily.nc')
    ds_concat.to_netcdf(filename)
    print("Saved:", filename)
            
    return ds_concat


# %%time
if __name__ == '__main__':
    ds_original = barra_daily(start_year="2017", end_year="2024", outdir="../data")

# Remove the 'timebands' variable (not sure why it got added)
ds = ds_original[['uas', 'vas']]

# Use the eastward and westward wind to calculate a speed and direction
speed = np.sqrt(ds["uas"]**2 + ds["vas"]**2)
speed_km_hr = speed * 3.6
direction = (270 - np.degrees(np.arctan2(ds["vas"], ds["uas"]))) % 360


# +
# Convert the wind direction into a categorical variable
def direction_to_compass(direction):
    compass_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    sector_width = 360 / len(compass_directions)  # 45° per sector
    index = np.round(direction / sector_width) % len(compass_directions)
    return np.array(compass_directions)[index.astype(int)]

compass = xr.apply_ufunc(direction_to_compass, direction)
# -

ds = ds.assign(speed=speed_km_hr, direction=direction, compass=compass)



list(ds.to_dataframe()[['compass', 'speed']].values)

dataset = ds

# +
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_wind_rose(dataset):
    """
    Create an enhanced wind rose diagram with 16 direction labels,
    nested speed distribution, and spacing between directional cones.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing 'speed' and 'direction' variables
    
    Returns:
    --------
    matplotlib.figure.Figure
        Wind rose plot
    """
    # Extract speed and direction
    speeds = dataset['speed'].values
    directions = dataset['direction'].values
    
    # Define speed bins (reversed to have min inside, max outside)
    speed_bins = [0, 2, 5, 10, 15, 20, 25, 30]
    
    # 16 direction sectors (22.5 degrees each)
    dir_bins = np.linspace(0, 360, 17)
    dir_labels = [
        'N', 'NNE', 'NE', 'ENE', 
        'E', 'ESE', 'SE', 'SSE', 
        'S', 'SSW', 'SW', 'WSW', 
        'W', 'WNW', 'NW', 'NNW'
    ]
    
    # Compute frequency of speeds in each direction
    freq_matrix = np.zeros((len(speed_bins)-1, len(dir_bins)-1))
    
    for i in range(len(speeds)):
        speed = speeds[i]
        direction = directions[i]
        
        # Find speed bin (reversed order)
        speed_bin_index = np.digitize(speed, speed_bins) - 1
        if speed_bin_index >= len(speed_bins) - 1:
            continue
        
        # Find direction bin
        dir_bin_index = np.digitize(direction, dir_bins) - 1
        if dir_bin_index >= len(dir_bins) - 1:
            dir_bin_index = 0
        
        freq_matrix[speed_bin_index, dir_bin_index] += 1
    
    # Normalize to percentages
    freq_matrix = freq_matrix / len(speeds) * 100
    
    # Create wind rose plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(12, 12))
    
    # Color palette (reversed to match speed bins)
    colors = plt.cm.viridis(np.linspace(1, 0, len(speed_bins)-1))
    
    # Width of each sector with spacing
    width = np.pi / 8 * 0.8  # 22.5 degrees, with 20% gap
    
    # Adjust angles for polar plot (0 at North, clockwise)
    theta = np.linspace(np.pi/2, -3*np.pi/2, 16, endpoint=False)
    
    # Plot each speed bin (from inside to outside)
    for i in range(len(speed_bins)-1):
        radii = freq_matrix[-(i+1), :]  # Reverse order of speed bins
        ax.bar(theta, radii, width=width, bottom=sum(freq_matrix[-(j+1),:] for j in range(i)), 
               color=colors[i], alpha=0.7, 
               label=f'{speed_bins[-(i+1)]}-{speed_bins[-(i)]} km/hr')
    
    # Customize plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(theta)
    ax.set_xticklabels(dir_labels, fontsize=8)
    ax.set_title('Enhanced Wind Rose Diagram', fontsize=15)
    
    # Legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), title='Wind Speed (km/hr)')
    
    plt.tight_layout()
    return fig


# Generate and plot
wind_rose_fig = create_wind_rose(ds)
plt.show()
# -


