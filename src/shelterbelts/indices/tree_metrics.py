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

# Create some layers for sheltered and unsheltered crop and grassland
ds['Grassland'] = (ds['worldcover'] == 30) & (~ds['all_combined'])  # Grassland and not a tree
ds['Cropland'] = (ds['worldcover'] == 40) & (~ds['all_combined'])  # Cropland and not a tree
ds['Water'] = (ds['worldcover'] == 80) & (~ds['all_combined'])  
ds['Other'] = (ds['worldcover'] != 30) & (ds['worldcover'] != 40) & (ds['worldcover'] != 80) & (~ds['all_combined'])  
ds['sheltered'] = ds['distance_to_shelterbelt'].notnull() # Within 100m of a shelterbelt in the windward direction
ds['production'] = ds['Grassland'] | ds['Cropland']
ds['unsheltered'] = ds['production'] & ~ds['sheltered']
ds['sheltered_grassland'] = (ds['sheltered'] & ds['Grassland'])
ds['sheltered_cropland'] = (ds['sheltered'] & ds['Cropland'])
ds['unsheltered_grassland'] = (ds['unsheltered'] & ds['Grassland'])
ds['unsheltered_cropland'] = (ds['unsheltered'] & ds['Cropland'])
ds['scattered_trees'] = (ds['shelterbelts'].astype(bool) & ~ds['shelter'])
ds['forest'] = ds['all_combined'] & ~ds['scattered_trees'] & ~ds['shelter_pruned3']

# +
# Calculate some stats about the region
total_pixels = int(xr.ones_like(ds['woodyveg_2024'], dtype=np.uint8).sum())
percent_trees = float(ds['all_combined'].sum() / total_pixels) * 100
percent_grass = float(ds['Grassland'].sum() / total_pixels) * 100
percent_crop = float(ds['Cropland'].sum() / total_pixels) * 100
percent_water = float(ds['Water'].sum() / total_pixels) * 100
percent_other = float(ds['Other'].sum() / total_pixels) * 100

# Stats about shelterbelts
total_trees = int(ds['all_combined'].sum())
percent_forest = float(ds['forest'].sum() / total_trees) * 100
percent_shelterbelt = float(ds['shelter_pruned3'].sum() / total_trees) * 100
percent_scattered = float(ds['scattered_trees'].sum() / total_trees) * 100

# Stats about sheltered farmland
total_grass = int(ds['Grassland'].sum())
total_crop = int(ds['Cropland'].sum())
percent_sheltered_grass = float(ds['sheltered_grassland'].sum() / total_grass) * 100
percent_unsheltered_grass = float(ds['unsheltered_grassland'].sum() / total_grass) * 100
percent_sheltered_crop = float(ds['sheltered_cropland'].sum() / total_crop) * 100
percent_unsheltered_crop = float(ds['unsheltered_cropland'].sum() / total_crop) * 100

# -

# Put the stats into tables
index = 'Percent landcover (%)'
df_total_stats = pd.DataFrame([
    {'trees': percent_trees, 'grassland': percent_grass, 'cropland': percent_crop, 'water':percent_water, 'other':percent_other}
], index=[index])
df_total_stats = df_total_stats.sort_values(by=index, axis=1, ascending=False)
df_total_stats = df_total_stats.round(2)
df_total_stats

index = 'Percent tree types (%)'
df_tree_stats = pd.DataFrame([
    {'forest': percent_forest, 'shelterbelt': percent_shelterbelt, 'scattered trees': percent_scattered}
], index=[index])
df_tree_stats = df_tree_stats.sort_values(by=index, axis=1, ascending=False)
df_tree_stats = df_tree_stats.round(2)
df_tree_stats

df_tree_stats = pd.DataFrame([
    {'grassland': percent_sheltered_grass, 'cropland': percent_sheltered_crop},
    {'grassland': percent_unsheltered_grass, 'cropland': percent_unsheltered_crop}
], index=['Percent sheltered (%)', 'Percent unsheltered (%)'])
df_tree_stats = df_tree_stats.round(2)
df_tree_stats



# +
# List the coords in each group
coord_lists = {i: list(zip(*np.where(shelterbelts == i))) for i in range(1, num_features)}

# Calculate area, width, length for each shelterbelt
group_stats = []
for i, coords in coord_lists.items():
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    
    area = len(coords)
    x_length = max(xs) - min(xs)
    y_length = max(ys) - min(ys)
    stats = {
        'area':area,
        'x_length':x_length,
        'y_length':y_length,
        'max_length':max(x_length, y_length)
        # 'coords':coords
    }
    group_stats.append(stats)

df_shelterbelts = pd.DataFrame(group_stats, index=coord_lists.keys())


# +
# Filter out any shelterbelts less than a certain length, e.g. 100m
length_threshold = 10   # 100m 
df_large_shelterbelts = df_shelterbelts[df_shelterbelts['max_length'] > length_threshold]
large_shelterbelts = shelterbelts.copy()
mask = ~np.isin(shelterbelts, df_large_shelterbelts.index)
large_shelterbelts[mask] = 0

# Label the large shelterbelts consecutively
unique_vals = np.unique(large_shelterbelts)
large_shelterbelts = np.searchsorted(unique_vals, large_shelterbelts)
# -

# Create DataArrays from the numpy arrays for adding to the xarray DataSet
da_shelterbelts = xr.DataArray(
    shelterbelts,
    dims=["y", "x"],
    coords={"x": ds.x, "y": ds.y},
    name="shelterbelts"
)
da_large_shelterbelts = xr.DataArray(
    large_shelterbelts,
    dims=["y", "x"],
    coords={"x": ds.x, "y": ds.y},
    name="large_shelterbelts"
)

# Add these groups to the original xarray
ds["shelterbelts"] = da_shelterbelts
ds["large_shelterbelts"] = da_large_shelterbelts

# +
# Plot the large shelterbelts
data = ds['large_shelterbelts']
masked_data = data.where(data > 0)  # Mask zeros and negatives

# Make non-tree pixels transparent
cmap = plt.cm.Set1
cmap.set_bad(color=(0, 0, 0, 0))  
masked_data.plot(cmap=cmap, vmin=1)
plt.show()

# +
