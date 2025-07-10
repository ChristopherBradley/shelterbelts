# +
# Download wind data from BARRA for a specific region, and create a wind rose based on data from 2017-2024

# Thredds page is here: https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f2551_3726_7908_8861
# Publication is here: http://www.bom.gov.au/research/publications/researchreports/BRR-097.pdf

# uas stands for Eastward Near-Surface Wind, and vas stands for Northward Near-Surface Wind

# +
import os
import argparse

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
# -

barra_abbreviations = {
    "uas": "Eastward Near-Surface Wind",
    "vas": "Northward Near-Surface Wind"
}


# +
# %%time
def barra_singlemonth_thredds(var="uas", latitude=-34.3890427, longitude=148.469499, buffer=0.01, year="2020", month="01"):
    
    url = f"https://thredds.nci.org.au/thredds/dodsC/ob53/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{year}{month}-{year}{month}.nc"

    try:
        ds = xr.open_dataset(url, engine="netcdf4")
    except Exception as e:
        # Likely no data for the specified month
        return None

    # This bbox selection was giving a 0 latitude coordinates, even with a larger slice, 
    # so I've made the code ignore the buffer and only return a single point for now.
    
    # bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    # ds_region = ds.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))
    # min_buffer_size = 0.03
    # if buffer < min_buffer_size:
    #     # Find a single point but keep the lat and lon dimensions for consistency
    #     ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')
    
    ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')

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

# Requires the 'pip install windrose' library
def wind_rose(ds, filename=None):
    """Uses the output from barra_daily to create a wind rose plot"""
    ds = ds.median(dim=['latitude', 'longitude'])
    ds = ds[['uas', 'vas']]
    speed = np.sqrt(ds["uas"]**2 + ds["vas"]**2)
    speed_km_hr = speed * 3.6
    direction = (270 - np.degrees(np.arctan2(ds["vas"], ds["uas"]))) % 360

    # The looks nice if the maximum direction occurs about 20% of the time
    y_ticks = range(0, 30, 10)   # Frequency percentage bins
    x_ticks = range(0, 40, 10)  # Wind speed magnitude bins
    title_fontsize = 20
    
    ax = WindroseAxes.from_ax()
    ax.bar(direction.values, speed_km_hr.values, bins=x_ticks, normed=True, nsector=8)
    ax.set_legend(
        title="Wind Speed (km/hr)"
    )
    ax.set_rgrids(y_ticks, labels=[f"{y}%" for y in y_ticks])
    ax.set_title("Wind Rose", fontsize=title_fontsize)

    if filename:
        plt.savefig(filename)
        print("Saved", filename)
        plt.close()
    else:
        plt.show()


def wind_dataframe(ds):
    """Create a dataframe of the frequency of each wind speed in each direction"""
    ds = ds.median(dim=['latitude', 'longitude'])
    ds = ds[['uas', 'vas']]
    speed = np.sqrt(ds["uas"]**2 + ds["vas"]**2)
    speed_km_hr = speed * 3.6
    direction = (270 - np.degrees(np.arctan2(ds["vas"], ds["uas"]))) % 360
    
    # Convert wind speed into a categorical variables
    speed_labels = "0-10km/hr", "10-20km/hr", "20-30km/hr", "30+ km/hr"
    speed_bins = [0,10,20,30]
    speed_binned = np.digitize(speed_km_hr, speed_bins) - 1
    ds["speed_binned"] = xr.DataArray(
        speed_binned,
        dims=["time"],
        coords={"time": ds.time},
        name="speed_binned"
    )
    
    # Convert the wind direction into a categorical variable
    compass_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    sector_width = 360 / len(compass_labels)  # 45° per sector
    direction_binned_da = np.round(direction / sector_width) % len(compass_labels)
    direction_binned = (direction_binned_da.values).astype(int)
    
    def direction_to_compass(direction_binned):
        return np.array(compass_labels)[direction_binned.astype(int)]
    
    compass = xr.apply_ufunc(direction_to_compass, direction_binned_da)
    ds["compass"] = compass
    
    # Create a matrix of the number of occurences of each speed and direction
    freq_matrix = np.zeros((len(speed_labels), len(compass_labels)))
    for s, d in zip(speed_binned, direction_binned):
        freq_matrix[s,d] += 1
        
    percentage_matrix = np.round(100 * freq_matrix/len(speed_binned), 2)
    df = pd.DataFrame(percentage_matrix, index=speed_labels, columns = compass_labels)

    max_speed = round(float(speed_km_hr.max()), 2)
    direction_max_speed = str(compass[speed_km_hr.argmax()].values)
    
    return df, max_speed, direction_max_speed

# # Usage example of wind_dataframe
# df, max_speed, max_direction = wind_dataframe(ds)

# # Calculating the direction with the most days with winds over 20km/hr
# df_20km_plus = df.loc['20-30km/hr'] + df.loc['30+ km/hr']
# direction_20km_plus = df_20km_plus.index[df_20km_plus.argmax()]


def barra_daily(variables=["uas", "vas"], lat=-34.3890427, lon=148.469499, buffer=0.01, start_year="2020", end_year="2021", outdir=".", stub="TEST", save_netcdf=True, plot=True):
    """Download 8day variables from BARRA at 4.4km resolution for the region/time of interest

    Parameters
    ----------
        variables: uas = Eastward Near Surface Wind, vas = Northward Near Surface Wind. See links at the top of this file for more details.
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory that the final .NetCDF gets saved.
        stub: The name to be prepended to each file download.
        save_netcdf: Boolean to save the final result to file.
        plot: Boolean to generate an output png image.
    
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
    ds = xr.merge(dss)

    ds = ds.drop_vars(['time_bnds', 'height', 'crs'])
    ds = ds.rename({'lat':'latitude', 'lon':'longitude'})

    if save_netcdf:
        filename = os.path.join(outdir, f'{stub}_barra_daily.nc')
        ds.to_netcdf(filename)
        print("Saved:", filename)

    if plot:
        filename = os.path.join(outdir, f"{stub}_barra_daily.png")    
        wind_rose(ds, filename)

    return ds

def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.1', help='Buffer in each direction in degrees (default is 0.1, or about 20kmx20km)')
    parser.add_argument('--start_year', default='2020', help='Inclusive, and the minimum start year is 1889. Setting the start and end year to the same value will get all data for that year.')
    parser.add_argument('--end_year', default='2021', help='Specifying a larger end_year than available will automatically give data up to the most recent date (currently 2025)')
    parser.add_argument('--outdir', default='.', help='The directory to save the outputs. (Default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--plot', default=False, action="store_true", help="Boolean to Save a png file that isn't geolocated, but can be opened in Preview. (Default: False)")

    return parser.parse_args()


# %%time
if __name__ == '__main__':
    args = parse_arguments()
    
    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    start_year = args.start_year
    end_year = args.end_year
    outdir = args.outdir
    stub = args.stub
    plot = args.plot
    
    barra_daily(["uas", "vas"], lat, lon, buffer, start_year, end_year, outdir, stub, save_netcdf=True, plot=plot)

