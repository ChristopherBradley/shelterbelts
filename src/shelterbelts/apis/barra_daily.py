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

# -

barra_abbreviations = {
    "uas": "Eastward Near-Surface Wind",
    "vas": "Northward Near-Surface Wind"
}


# +
# %%time
def barra_singlemonth(var="uas", latitude=-34.389, longitude=148.469, buffer=0.01, year="2020", month="01", gdata=False, temporal='day'):
    temporals = ['20min', '1hr', 'day', 'mon'] # 3hr doesn't have uas

    if gdata:
        # Needs to be run on NCI with access to the ob53 project
        url = f"/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/{temporal}/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_{temporal}_{year}{month}-{year}{month}.nc"
    else:
        # URL for NCI thredds - should work anywhere with internet access
        url = f"https://thredds.nci.org.au/thredds/dodsC/ob53/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/{temporal}/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_{temporal}_{year}{month}-{year}{month}.nc"

    try:
        ds = xr.open_dataset(url, engine="netcdf4")
    except Exception as e:
        # Likely no data for the specified month
        return None

    # This bbox selection was giving a 0 latitude coordinates, even with a larger slice, so I've made the code ignore the buffer and only return a single point for now.
    # Issue in May 2025, working again on 7 August, then broken again on 3 September: 
    ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')
    
    # bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    # ds_region = ds.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))
    # min_buffer_size = 0.03
    # if buffer < min_buffer_size:
    #     # Find a single point but keep the lat and lon dimensions for consistency
    #     ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')
    
    return ds_region

# barra_singlemonth_thredds()


# +
# %%time
def barra_single_year(var="uas", latitude=-34.389, longitude=148.469, buffer=0.01, year="2020", gdata=False, temporal='day'):
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    dss = []
    for month in months:
        ds_month = barra_singlemonth(var, latitude, longitude, buffer, year, month, gdata, temporal)
        if ds_month:
            dss.append(ds_month)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat
    
# barra_single_year()
# Took 5 secs


# +
def barra_multiyear(var="uas", latitude=-34.389, longitude=148.469, buffer=0.01, years=["2020", "2021"], gdata=False, temporal='day'):
    dss = []
    for year in years:
        ds_year = barra_single_year(var, latitude, longitude, buffer, year, gdata, temporal)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat

# barra_multiyear()


# -

# Requires the 'pip install windrose' library
def wind_rose(ds, filename=None):
    """Uses the output from barra_daily to create a wind rose plot"""
    from windrose import WindroseAxes  # Putting this inside the function so the rest of the file can still be used even if windrose hasn't been installed

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


def dominant_wind_direction(ds, threshold_kmh=15):
    """Return the compass direction with the most days above a wind speed threshold"""
    compass_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    ds = ds.median(dim=['latitude', 'longitude'])
    uas, vas = ds["uas"], ds["vas"]
    
    # Calculate wind speed in km/h
    speed = np.sqrt(uas**2 + vas**2) * 3.6

    # Calculate wind direction in degrees (meteorological convention)
    direction = (270 - np.degrees(np.arctan2(vas, uas))) % 360

    # Filter time steps with speed above the threshold
    mask = speed > threshold_kmh
    direction = direction.where(mask, drop=True)

    # Bin direction into compass sectors
    sector_width = 360 / len(compass_labels)
    direction_binned = np.round(direction / sector_width) % len(compass_labels)
    direction_binned = direction_binned.astype(int)

    # Count occurrences per direction
    counts = pd.Series(direction_binned.values).value_counts().sort_index()
    direction_counts = pd.Series(0, index=np.arange(len(compass_labels)))
    direction_counts.update(counts)

    # Get the direction with the most days above threshold
    most_frequent_index = direction_counts.idxmax()
    most_frequent_direction = compass_labels[most_frequent_index]

    df_direction_counts = df = pd.DataFrame({
        "Direction": compass_labels,
        "Count": direction_counts.values
    })
    
    return most_frequent_direction, df_direction_counts


# +
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
    direction_max_speed = str(compass[speed_km_hr.argmax(dim=...)].values)
    
    return df, max_speed, direction_max_speed

# Use example =
# df, max_speed, max_direction = wind_dataframe(ds)
# df_20km_plus = df.loc['20-30km/hr'] + df.loc['30+ km/hr']
# direction_20km_plus = df_20km_plus.index[df_20km_plus.argmax()]


# -

def barra_daily(variables=["uas", "vas"], lat=-34.389, lon=148.469, buffer=0.01, start_year="2020", end_year="2021", outdir=".", stub="TEST", save_netcdf=True, plot=True, gdata=False, temporal='day'):
    """Download 8day variables from BARRA at 4.4km resolution for the region and time of interest

    Parameters
    ----------
    variables : list of str, optional
        Default is ["uas", "vas"] for Eastward and Northward Near-Surface Wind, respectively. 
        See links at the top of this file for more details.
    lat : float, optional
        Latitude in WGS 84 (EPSG:4326). Default is -34.389.
    lon : float, optional
        Longitude in WGS 84 (EPSG:4326). Default is 148.469.
    buffer : float, optional
        -- Note: Buffer option is currently disabled due to a bug in the API, so we just get the nearest point.
    start_year : str, optional
        Start year (inclusive). The minimum available year is 1889.
        Default is "2020".
    end_year : str, optional
        End year (inclusive). Data beyond the available range will be capped
        at the most recent available date. Default is "2021".
    outdir : str, optional
        Output directory for saving results. Default is the current directory.
    stub : str, optional
        Prefix for output filenames. Default is "TEST".
    save_netcdf : bool, optional
        Whether to save results as a NetCDF file. Default is True.
    plot : bool, optional
        Whether to generate a wind rose visualisation (PNG). Default is True.
    gdata : bool, optional
        Whether to access data via NCI /g/data/xe2 path (requires NCI access).
        If False, uses public THREDDS server. Default is False.
    temporal : str, optional
        Temporal resolution of the data. Options are '20min', '1hr', 'day', 'mon'.
        Default is 'day'.

    Returns
    -------
    xarray.Dataset
        Dataset containing the requested variables with dimensions for time,
        latitude, and longitude.

    Notes
    -----
    When ``save_netcdf=True``, it writes:
    ``{stub}_barra_daily.nc``

    When ``plot=True``, it writes:
    ``{stub}_barra_daily.png`` (a wind rose visualisation)

    Examples
    --------
    Download wind data with default parameters:

    >>> ds = barra_daily(save_netcdf=False, plot=False)
    >>> 'uas' in ds.data_vars and 'vas' in ds.data_vars  # Eastward and northward wind speed
    True
    
    Visualising the wind rose:

    .. plot::

        from shelterbelts.apis.barra_daily import barra_daily, wind_rose
        
        ds = barra_daily(save_netcdf=False, plot=False)
        wind_rose(ds)

    """
    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = barra_multiyear(variable, lat, lon, buffer, years, gdata, temporal)
        dss.append(ds_variable)
    ds = xr.merge(dss, compat="override")

    vars_to_drop = ['time_bnds', 'height', 'crs']
    ds = ds.drop_vars([v for v in vars_to_drop if v in ds])
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
    
    parser.add_argument('--lat', default=-34.389, type=float, help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default=148.469, type=float, help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default=0.01, type=float, help='Buffer in each direction in degrees (default is 0.01, or about 2kmx2km)')
    parser.add_argument('--start_year', default='2020', help='Inclusive, and the minimum start year is 1889. Setting the start and end year to the same value will get all data for that year.')
    parser.add_argument('--end_year', default='2021', help='Specifying a larger end_year than available will automatically give data up to the most recent date (currently 2025)')
    parser.add_argument('--outdir', default='.', help='The directory to save the outputs. (Default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--no-save-netcdf', dest='save_netcdf', action="store_false", default=True, help='Disable saving NetCDF output (default: enabled)')
    parser.add_argument('--no-plot', dest='plot', action="store_false", default=True, help='Disable PNG visualisation (default: enabled)')
    parser.add_argument('--gdata', action="store_true", default=False, help='Access data via NCI /g/data path (requires NCI access). Default: False')
    parser.add_argument('--temporal', default='day', help='Temporal resolution of the data. Options are 20min, 1hr, day, mon. Default: day')

    return parser


# %%time
if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    
    barra_daily(["uas", "vas"], args.lat, args.lon, args.buffer, args.start_year, args.end_year, args.outdir, args.stub, save_netcdf=args.save_netcdf, plot=args.plot, gdata=args.gdata, temporal=args.temporal)

# +
# In my experiments comparing thredds and gdata with the default arguments, these were the times taken.
# thredds: 21 secs, then 10, 10, 10
# gdata: 6 secs, then 4, 4, 4

# +
# # %%time
# Performance comparison of different temporal requests
# ds = barra_daily(gdata=True, save_netcdf=False, plot=False, temporal='mon') 
# # For monthly: 5 secs, then 1, 1, 1
# # For daily: 13 secs, then 3, 3, 3
# # For 1hour: 2 mins, then 10 secs
# # For 20min: 2 mins then error with drop_vars

# +
# # %%time

# # var = 'uas'
# # year = '2020'
# # month = '01'
# # url = f"/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{year}{month}-{year}{month}.nc"

# latitude=-34.3890427
# longitude=148.469499
# url = '/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day/uas/latest/uas_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_202001-202001.nc'
# ds = xr.open_dataset(url, engine="netcdf4")
# ds_region = ds.sel(lat=[latitude], lon=[longitude], method='nearest')    

# url = '/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/mon/uas/latest/uas_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_mon_202505-202505.nc'
# ds_month = xr.open_dataset(url, engine="netcdf4")
# ds_month_region = ds_month.sel(lat=[latitude], lon=[longitude], method='nearest')    

# url = '/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/20min/uas/latest/uas_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_20min_202505-202505.nc'
# ds_20min = xr.open_dataset(url, engine="netcdf4")
# ds_20min_region = ds_20min.sel(lat=[latitude], lon=[longitude], method='nearest')    

# monthly = ds_month_region['uas'].values[0, 0, 0]
# min20 = ds_20min_region['uas'].values[:, 0, 0]
# daily = ds_region['uas'].values[:, 0, 0]  # shape (31,)

# print("Monthly value:", monthly)

# print("20min mean:", np.mean(min20))
# print("20min median:", np.median(min20))
# print("20min max:", np.max(min20))
# print("20min min:", np.min(min20))
# print("20min std:", np.std(min20))

# print("Daily mean:", np.mean(daily))
# print("Daily median:", np.median(daily))
# print("Daily max:", np.max(daily))
# print("Daily min:", np.min(daily))
# print("Daily std:", np.std(daily))
