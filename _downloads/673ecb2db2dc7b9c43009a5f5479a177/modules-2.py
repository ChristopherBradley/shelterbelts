from shelterbelts.apis.barra_daily import barra_daily, wind_rose

ds = barra_daily(save_netcdf=False, plot=False)
wind_rose(ds)