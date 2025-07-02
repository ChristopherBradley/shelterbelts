# Load worldcover and reproject match
filename = "data/Fulham_worldcover.tif"
da_worldcover = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
da_worldcover_matched = da_worldcover.rio.reproject_match(da_original)
ds["worldcover"] = da_worldcover_matched
ds["worldcover_veg"] = (ds["worldcover"] == 10)

%%time
# Load canopy_height and reproject match
filename = "data/Fulham_canopy_height.tif"
da = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
da_matched = da.rio.reproject_match(ds)
da_matched = da_matched.where(da_matched != 255, np.nan)
ds["canopy_height"] = da_matched
ds['canopy_height_veg'] = (ds["canopy_height"] >= 1)

%%time
# Load all 5 years of woody veg to see what's changed
# Based on visual inspection, I think the 2021 raster overpredicts vegetation, so leaving it out
years = ["2019", "2020", "2022", "2023", "2024"]
for year in years:
    filename = f'/Users/christopherbradley/Documents/PHD/Data/Annual_woody_vegetation_and_canopy_cover_grids_for_Tasmania-z_BE-P62-/data/WoodyVeg/Tas_WoodyVeg_{year}03_v2.2.tif'
    da_original = rxr.open_rasterio(filename).isel(band=0).drop_vars('band')
    da = da_original.sel(x=slice(minx, maxx), y=slice(miny, maxy))
    da = (da.where(da != 255, 1) - 1).astype(bool)  # Convert NaN and no tree to False, and tree to True
    ds[f"woodyveg_{year}"] = da

# Merge all the vegetation layers into 1 (since they usually underpredict vegetation rather than overpredict)
ds["woodyveg_combined"] = ds["woodyveg_2019"] 
for year in years:
    ds["woodyveg_combined"] = ds["woodyveg_combined"] | ds[f"woodyveg_{year}"]
ds["all_combined"] = ds['worldcover_veg'] | ds['canopy_height_veg'] | ds["woodyveg_combined"] 
ds["all_combined"].plot()