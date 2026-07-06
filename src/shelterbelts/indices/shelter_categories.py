import os
import argparse

import numpy as np
import xarray as xr
import rioxarray as rxr
import scipy

from shelterbelts.apis.barra_daily import wind_dataframe, dominant_wind_direction
from shelterbelts.apis.worldcover import tif_categorical, visualise_categories
from shelterbelts.indices.patch_metrics import linear_categories_labels, linear_categories_cmap


# Sheltered farmland is encoded based on the tree category (10-19) providing the shelter,
    # using a matching (30-39) label for grassland and (40-49) for cropland.
    # However, the density-based shelter (instead of distance-based) just uses 30 or 40 as unsheltered, and 32 or 42 as sheltered.

_tree_source_labels = {
    3: 'Patch Edge',
    4: 'Other Trees',
    5: 'Trees in Gullies',
    6: 'Trees on Ridges',
    7: 'Trees next to Roads',
    8: 'Linear Patches',
    9: 'Non-linear Patches',
}
_grassland_labels = {30: 'Unsheltered Grassland', 31: 'Unsheltered Grassland', 32: 'Sheltered Grassland'}
_cropland_labels = {40: 'Unsheltered Cropland', 41: 'Unsheltered Cropland', 42: 'Sheltered Cropland'}
for _digit, _name in _tree_source_labels.items():
    _grassland_labels[30 + _digit] = f'Grassland sheltered by {_name}'
    _cropland_labels[40 + _digit] = f'Cropland sheltered by {_name}'


def _interp_cmap(start_rgb, end_rgb, codes):
    """Linearly interpolate an RGB gradient across a list of integer codes."""
    n = len(codes)
    cmap = {}
    for i, code in enumerate(codes):
        t = i / (n - 1) if n > 1 else 0.0
        cmap[code] = tuple(int(round(s + (e - s) * t)) for s, e in zip(start_rgb, end_rgb))
    return cmap


# Yellow/olive gradient for grassland shelter, pink/purple gradient for cropland shelter
_grassland_cmap = _interp_cmap((255, 255, 153), (102, 102, 0), list(range(30, 40)))
_cropland_cmap = _interp_cmap((255, 204, 229), (102, 0, 102), list(range(40, 50)))
_grassland_cmap[30] = (255, 255, 76)   # worldcover grassland base colour
_cropland_cmap[40] = (240, 150, 255)   # worldcover cropland base colour
_grassland_cmap[31] = (0, 0, 0)        # legacy 'Unsheltered Grassland' from the old version
_cropland_cmap[41] = (0, 0, 0)         # legacy 'Unsheltered Cropland' from the old version

shelter_categories_labels = linear_categories_labels | _grassland_labels | _cropland_labels
shelter_categories_cmap = linear_categories_cmap | _grassland_cmap | _cropland_cmap


direction_map = {
    'N':  (-1,  0),
    'NE': (-1,  1),
    'E':  ( 0,  1),
    'SE': ( 1,  1),
    'S':  ( 1,  0),
    'SW': ( 1, -1),
    'W':  ( 0, -1),
    'NW': (-1, -1),
}
inverted_direction_map = {v: k for k, v in direction_map.items()}

# One distinct colour per compass direction for MULTI_LAYER tifs (clockwise from N)
_direction_colours = {
    'N':  (  0,  80, 255),  # Blue
    'NE': (  0, 200, 255),  # Cyan
    'E':  (  0, 180,   0),  # Green
    'SE': (180, 180,   0),  # Yellow
    'S':  (255, 128,   0),  # Orange
    'SW': (220,   0,   0),  # Red
    'W':  (200,   0, 200),  # Magenta
    'NW': ( 90,   0, 220),  # Purple
}


def compute_distance_to_tree_TH(shelter_heights, wind_dir='E', max_distance=20):
    """For each non-tree pixel, find the distance to the nearest upwind tree tall enough to shelter it.
    
    A tree of height h shelters pixels up to h pixels downwind, rounded to the nearest integer.
    Returns NaN for tree pixels and unsheltered pixels.
    """
    dy, dx = direction_map[wind_dir]
    step_length = (dy ** 2 + dx ** 2) ** 0.5  # 1 for cardinal directions, √2 for diagonals
    distances = xr.full_like(shelter_heights, np.nan)
    is_tree = shelter_heights > 0

    for d in range(1, max_distance + 1):
        euclidean_distance = round(d * step_length)  # Euclidean distance in pixels, rounded to nearest integer
        shifted = shelter_heights.shift(x=dx * d, y=dy * d, fill_value=0)
        can_shelter = shifted >= euclidean_distance
        unassigned = ~is_tree & distances.isnull()
        distances = xr.where(unassigned & can_shelter, float(euclidean_distance), distances)

    return distances


def compute_distance_and_source(shelter_heights, tree_source, wind_dir='E', max_distance=100):
    """Like compute_distance_to_tree_TH, but also records which tree category provides the shelter.

    For each non-tree pixel, walks upwind until it finds a tree tall enough to shelter it, then
    records both the (rounded Euclidean) distance in pixels and the category of that sheltering tree.
    """
    dy, dx = direction_map[wind_dir]
    step_length = (dy ** 2 + dx ** 2) ** 0.5  # 1 for cardinal directions, √2 for diagonals
    distances = xr.full_like(shelter_heights, np.nan)
    sources = xr.full_like(shelter_heights, np.nan)
    is_tree = shelter_heights > 0

    for d in range(1, max_distance + 1):
        euclidean_distance = round(d * step_length)  # Euclidean distance in pixels, rounded to nearest integer
        shifted = shelter_heights.shift(x=dx * d, y=dy * d, fill_value=0)
        shifted_src = tree_source.shift(x=dx * d, y=dy * d, fill_value=0)
        can_shelter = shifted >= euclidean_distance
        assign = ~is_tree & distances.isnull() & can_shelter
        distances = xr.where(assign, float(euclidean_distance), distances)
        sources = xr.where(assign, shifted_src, sources)

    return distances, sources


def _combine_min_source(d1, s1, d2, s2):
    """Combine two (distance, source) pairs, keeping the source of the smaller positive distance."""
    take1 = d1.fillna(np.inf) <= d2.fillna(np.inf)
    return xr.where(take1, d1, d2), xr.where(take1, s1, s2)


def compute_tree_densities(tree_percent, min_distance=0, max_distance=20, mask_adjacencies=False):
    """Using a boolean tree mask, assign a percentage density to each pixel within the given distance"""
    tree_mask = tree_percent > 0

    structuring_element = np.ones((3, 3))  # This defines adjacency (including diagonals)
    adjacent_mask = scipy.ndimage.binary_dilation(tree_mask, structure=structuring_element)

    # Calculate the number of trees in a donut between the inner and outer circle
    y, x = np.ogrid[-max_distance:max_distance+1, -max_distance:max_distance+1]
    kernel = (x**2 + y**2 <= max_distance**2) & (x**2 + y**2 >= min_distance**2)
    kernel = kernel.astype(float)

    total_tree_cover = scipy.signal.fftconvolve(tree_percent, kernel, mode='same')
    percent_trees = (total_tree_cover / kernel.sum()) * 100

    # Mask out trees
    percent_trees[np.where(tree_mask)] = np.nan
    percent_trees[percent_trees < 1] = 0

    if mask_adjacencies:
        # We probably don't want to mask adjacencies when estimating amount of sheltered farmland in Australia
        # We probably do want to mask adjacencies when analysing the effects of tree cover on nearby productivity
        percent_trees[np.where(adjacent_mask)] = np.nan

    da_percent_trees = xr.DataArray(
        percent_trees,
        dims=("y", "x"),
        coords={"y": tree_percent.coords["y"], "x": tree_percent.coords["x"]},
        name="percent_trees"
    )

    return da_percent_trees


def _label_farmland(da, grassland, cropland, sheltered, source_digit):
    """Re-encode grassland/cropland by shelter status.

    If source_digit is provided (wind methods), sheltered farmland is 3X/4X per the sheltering tree.
    If source_digit is None (density / MULTI_LAYER), sheltered farmland is the generic 32/42.
    """
    da_out = da.copy()
    if source_digit is None:
        sheltered_grass_code = 32
        sheltered_crop_code = 42
    else:
        sheltered_grass_code = 30 + source_digit
        sheltered_crop_code = 40 + source_digit
    da_out = xr.where(grassland & sheltered, sheltered_grass_code, da_out)
    da_out = xr.where(grassland & ~sheltered, 30, da_out)
    da_out = xr.where(cropland & sheltered, sheltered_crop_code, da_out)
    da_out = xr.where(cropland & ~sheltered, 40, da_out)
    return da_out.astype('uint8').rio.write_crs(da.rio.crs)


def shelter_categories(linear_data, wind_data=None, height_tif=None, outdir='.', stub='TEST',
                       wind_method='WINDWARD', wind_threshold=20, distance_threshold=20,
                       density_threshold=5, savetif=True, plot=True, crop_pixels=None, debug=False):
    """Label sheltered farmland, and the type of tree providing the shelter (for non-density methods).

    Parameters
    ----------
    linear_data : str, xarray.Dataset, or xarray.DataArray
        Integer tif file generated by patch_metrics.py, or a Dataset/DataArray containing the 'linear_categories' band. 
    wind_data : str or xarray.Dataset, optional
        NetCDF with eastward and northward wind speed generated by barra_daily.py. 
        If not provided then sheltered/unsheltered is defined by nearby tree density instead.
    height_tif : str, optional
        Integer tif file generated by canopy_height.py. 
        If provided, distance_threshold is measured in tree heights rather than pixels.
    outdir : str, optional
        Output directory for saving results.
    stub : str, optional
        Prefix for output filenames.
    wind_method : str, optional
        Either 'WINDWARD', 'MOST_COMMON', 'MAX', 'HAPPENED', 'ANY', or 'MULTI_LAYER'.
        MOST_COMMON assumes only downwind shelter, using the most common wind direction above the wind_threshold.
        WINDWARD assumes downwind shelter to distance_threshold, and upwind shelter to (distance_threshold / 2).
        MAX assumes shelter from the direction of maximum wind speed.
        HAPPENED refers to any direction where the winds exceeded the wind_threshold.
        ANY refers to all 8 compass directions.
        MULTI_LAYER saves an 8-band GeoTIFF (bands clockwise: N, NE, E, SE, S, SW, W, NW). Does not require wind_data.
    wind_threshold : int, optional
        Wind speed threshold in km/h.
    distance_threshold : int, optional
        Distance from trees that counts as sheltered (pixels, or tree heights when height_tif is given).
    density_threshold : int, optional
        Percentage tree cover within distance_threshold that counts as sheltered (only applies if wind_data is not provided).
    savetif : bool, optional
        Whether to save the results as a GeoTIFF.
    plot : bool, optional
        Whether to generate a PNG visualisation.
    crop_pixels : int, optional
        Number of pixels to crop from each edge of the output.

    Returns
    -------
    xarray.Dataset
        Dataset containing the 'shelter_categories' band, where the integers represent the categories
        defined in 'shelter_categories_labels'.

    Notes
    -------
    When savetif=True, it outputs a GeoTIFF file with embedded color map:
    {stub}_shelter_categories.tif

    When plot=True, it outputs a PNG visualisation with legend:
    {stub}_shelter_categories.png

    Examples
    --------
    Using a file path as input:

    >>> from shelterbelts.utils.filepaths import get_filename
    >>> linear_file = get_filename('g2_26729_linear_categories.tif')
    >>> wind_file = get_filename('g2_26729_barra_daily.nc')
    >>> ds = shelter_categories(linear_file, wind_data=wind_file, outdir='/tmp', plot=False, savetif=False)
    >>> 'shelter_categories' in set(ds.data_vars)
    True

    Here's how different parameters affect the shelter categorisation:

    .. plot::

        import rioxarray as rxr
        from shelterbelts.indices.shelter_categories import shelter_categories, shelter_categories_cmap, shelter_categories_labels
        from shelterbelts.utils.filepaths import get_filename
        from shelterbelts.utils.visualisation import visualise_categories_sidebyside

        linear_file = get_filename('g2_26729_linear_categories.tif')
        wind_file = get_filename('g2_26729_barra_daily.nc')
        da = rxr.open_rasterio(linear_file).isel(band=0).drop_vars('band')
        ds_linear = da.to_dataset(name='linear_categories')

        # density_threshold: 3 vs 10 (density method, no wind data)
        ds1 = shelter_categories(ds_linear, outdir='/tmp', stub='dens1', plot=False, savetif=False, density_threshold=3)
        ds2 = shelter_categories(ds_linear, outdir='/tmp', stub='dens2', plot=False, savetif=False, density_threshold=10)
        visualise_categories_sidebyside(
            ds1['shelter_categories'], ds2['shelter_categories'],
            colormap=shelter_categories_cmap, labels=shelter_categories_labels,
            title1="density_threshold=3", title2="density_threshold=10"
        )

        # wind_method: MOST_COMMON vs WINDWARD
        ds1 = shelter_categories(ds_linear, wind_data=wind_file, outdir='/tmp', stub='wind1', plot=False, savetif=False, wind_method='MOST_COMMON')
        ds2 = shelter_categories(ds_linear, wind_data=wind_file, outdir='/tmp', stub='wind2', plot=False, savetif=False, wind_method='WINDWARD')
        visualise_categories_sidebyside(
            ds1['shelter_categories'], ds2['shelter_categories'],
            colormap=shelter_categories_cmap, labels=shelter_categories_labels,
            title1="wind_method=MOST_COMMON", title2="wind_method=WINDWARD"
        )

        # distance_threshold: 10 vs 30 (with wind data)
        ds1 = shelter_categories(ds_linear, wind_data=wind_file, outdir='/tmp', stub='dist1', plot=False, savetif=False, distance_threshold=10, wind_method='MOST_COMMON')
        ds2 = shelter_categories(ds_linear, wind_data=wind_file, outdir='/tmp', stub='dist2', plot=False, savetif=False, distance_threshold=30, wind_method='MOST_COMMON')
        visualise_categories_sidebyside(
            ds1['shelter_categories'], ds2['shelter_categories'],
            colormap=shelter_categories_cmap, labels=shelter_categories_labels,
            title1="distance_threshold=10", title2="distance_threshold=30"
        )

        # wind_threshold: 10 vs 30 (km/h)
        ds1 = shelter_categories(ds_linear, wind_data=wind_file, outdir='/tmp', stub='wt1', plot=False, savetif=False, wind_threshold=10, wind_method='MOST_COMMON')
        ds2 = shelter_categories(ds_linear, wind_data=wind_file, outdir='/tmp', stub='wt2', plot=False, savetif=False, wind_threshold=30, wind_method='MOST_COMMON')
        visualise_categories_sidebyside(
            ds1['shelter_categories'], ds2['shelter_categories'],
            colormap=shelter_categories_cmap, labels=shelter_categories_labels,
            title1="wind_threshold=10", title2="wind_threshold=30"
        )
    """
    if isinstance(linear_data, xr.Dataset):
        ds_input = linear_data.copy(deep=True)
        band = 'linear_categories' if 'linear_categories' in ds_input.data_vars else 'tree_categories'
        da = ds_input[band]
    elif isinstance(linear_data, xr.DataArray):
        da = linear_data
        ds_input = None
    else:
        da = rxr.open_rasterio(linear_data).squeeze('band').drop_vars('band')
        ds_input = None

    # Accept both the current (30/40) and legacy (31/32, 41/42) cover encodings
    grassland = da.isin([30, 31, 32])
    cropland = da.isin([40, 41, 42])

    # Only trees 12-19 block the wind (scattered trees 11 do not)
    shelter = (da >= 12) & (da < 20)
    tree_source = xr.where(shelter, da, 0)

    # Any tree pixel (including scattered) counts towards percentage tree cover for the density method
    tree_mask = (da >= 10) & (da < 20)
    tree_percent = tree_mask.astype(float)

    if height_tif:
        da_heights = rxr.open_rasterio(height_tif).squeeze('band').drop_vars('band')
        da_heights_reprojected = da_heights.rio.reproject_match(da)
        da_heights_nan = da_heights_reprojected.where(shelter, np.nan)
        shelter_heights = da_heights_nan.clip(min=0, max=60)
        pixel_size = 10  # metres
        shelter_heights = (shelter_heights / pixel_size) * distance_threshold
    else:
        shelter_heights = shelter.where(shelter, other=np.nan) * distance_threshold

    if isinstance(wind_data, xr.Dataset):
        ds_wind = wind_data
    elif wind_data:
        ds_wind = xr.load_dataset(wind_data)
    else:
        ds_wind = None

    # MULTI_LAYER: compute shelter distances independently for all 8 directions, return as 8-band tif
    if wind_method == 'MULTI_LAYER':
        max_distance = 100 if height_tif else distance_threshold
        directions = list(direction_map.keys())
        distance_rasters = [
            compute_distance_to_tree_TH(shelter_heights, d, max_distance)
                .fillna(0).clip(0, 255).astype('uint8')
            for d in directions
        ]
        da_multi = xr.concat(distance_rasters, dim='direction').assign_coords(direction=directions)
        da_multi = da_multi.rio.write_crs(da.rio.crs)

        sheltered = (da_multi > 0).any(dim='direction')
        da_shelter_categories = _label_farmland(da, grassland, cropland, sheltered, None)

        if crop_pixels is not None and crop_pixels != 0:
            da_multi = da_multi.isel(x=slice(crop_pixels, -crop_pixels), y=slice(crop_pixels, -crop_pixels))
            da_shelter_categories = da_shelter_categories.isel(x=slice(crop_pixels, -crop_pixels), y=slice(crop_pixels, -crop_pixels))

        ds = ds_input if ds_input is not None else da.to_dataset(name='linear_categories')
        ds['shelter_categories'] = da_shelter_categories
        ds['shelter_distances'] = da_multi

        if savetif:
            for direction in directions:
                band_da = da_multi.sel(direction=direction).rio.write_nodata(0)
                colour = _direction_colours[direction]
                colormap = {0: (0, 0, 0)} | {v: colour for v in range(1, 256)}
                band_filename = os.path.join(outdir, f"{stub}_shelter_{direction}.tif")
                tif_categorical(band_da, band_filename, colormap)
            combined_filename = os.path.join(outdir, f"{stub}_shelter_categories.tif")
            da_multi.rio.to_raster(combined_filename)
            print(f"Saved: {combined_filename}")

        if plot:
            import matplotlib.pyplot as plt
            _, axes = plt.subplots(2, 4, figsize=(24, 11))
            cmaps = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'YlOrBr', 'GnBu', 'PuRd']
            for ax, direction, d_raster, cmap_name in zip(axes.flat, directions, distance_rasters, cmaps):
                ax.imshow(d_raster.values, cmap=cmap_name)
                ax.set_title(direction, fontsize=30)
                ax.axis('off')
            plt.tight_layout()
            filename_png = os.path.join(outdir, f"{stub}_shelter_categories.png")
            plt.savefig(filename_png, dpi=150, bbox_inches='tight')
            plt.close()

        return ds

    max_search = 100  # Trees define their own sheltering distance via shelter_heights
    source_digit = None
    da_distance_or_percent = None

    if ds_wind is not None:
        if wind_method == 'MAX':
            _, _, primary_wind_direction = wind_dataframe(ds_wind)
            distances, sources = compute_distance_and_source(shelter_heights, tree_source, primary_wind_direction, max_search)

        elif wind_method == 'MOST_COMMON':
            primary_wind_direction, _ = dominant_wind_direction(ds_wind, wind_threshold)
            distances, sources = compute_distance_and_source(shelter_heights, tree_source, primary_wind_direction, max_search)

        elif wind_method == 'WINDWARD':
            primary_wind_direction, _ = dominant_wind_direction(ds_wind, wind_threshold)
            d1, s1 = compute_distance_and_source(shelter_heights, tree_source, primary_wind_direction, max_search)

            windward_scaling_factor = 0.5
            opposite_wind_direction = inverted_direction_map[tuple(np.array(direction_map[primary_wind_direction]) * -1)]
            d2, s2 = compute_distance_and_source(shelter_heights * windward_scaling_factor, tree_source, opposite_wind_direction, max_search)

            distances, sources = _combine_min_source(d1, s1, d2, s2)

        elif wind_method == 'HAPPENED':
            _, df_wind = dominant_wind_direction(ds_wind, wind_threshold)
            strong_wind_directions = list(df_wind.loc[df_wind['Count'] > 0, 'Direction'])
            distances, sources = None, None
            for wind_direction in strong_wind_directions:
                di, si = compute_distance_and_source(shelter_heights, tree_source, wind_direction, max_search)
                distances, sources = (di, si) if distances is None else _combine_min_source(distances, sources, di, si)
            if distances is None:  # Wind never exceeded the threshold, so nothing needs shelter
                distances = xr.full_like(shelter_heights, np.nan)
                sources = xr.full_like(shelter_heights, np.nan)

        elif wind_method == 'ANY':
            distances, sources = None, None
            for wind_direction in direction_map:
                di, si = compute_distance_and_source(shelter_heights, tree_source, wind_direction, max_search)
                distances, sources = (di, si) if distances is None else _combine_min_source(distances, sources, di, si)

        else:
            raise ValueError(f"Unsupported wind_method '{wind_method}'.")

        sheltered = distances > 0
        source_digit = sources.fillna(0).astype(int) % 10  # 2-9 for the sheltering tree category
        da_distance_or_percent = distances
        filename_distance_or_density = os.path.join(outdir, f"{stub}_shelter_distances.tif")

    else:
        # Density method based on percent tree cover nearby, rather than distance from trees.
        da_percent_trees = compute_tree_densities(tree_percent, max_distance=distance_threshold)
        sheltered = da_percent_trees >= density_threshold
        da_distance_or_percent = da_percent_trees
        filename_distance_or_density = os.path.join(outdir, f"{stub}_shelter_densities.tif")

    da_shelter_categories = _label_farmland(da, grassland, cropland, sheltered, source_digit)
    da_distance_or_percent = da_distance_or_percent.rio.write_crs(da.rio.crs)  # Some methods lose the crs due to xr.where()

    if crop_pixels is not None and crop_pixels != 0:
        da_shelter_categories = da_shelter_categories.isel(x=slice(crop_pixels, -crop_pixels), y=slice(crop_pixels, -crop_pixels))
        da_distance_or_percent = da_distance_or_percent.isel(x=slice(crop_pixels, -crop_pixels), y=slice(crop_pixels, -crop_pixels))

    ds = ds_input if ds_input is not None else da.to_dataset(name='linear_categories')
    ds['shelter_categories'] = da_shelter_categories

    if savetif:
        filename = os.path.join(outdir, f"{stub}_shelter_categories.tif")
        tif_categorical(ds['shelter_categories'], filename, shelter_categories_cmap)
        da_distance_or_percent.fillna(0).astype('uint8').rio.to_raster(filename_distance_or_density)
        print(f"Saved: {filename_distance_or_density}")

    if plot:
        filename_png = os.path.join(outdir, f"{stub}_shelter_categories.png")
        visualise_categories(ds['shelter_categories'], filename_png, shelter_categories_cmap, shelter_categories_labels, "Shelter Categories")

    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()

    parser.add_argument('linear_data', help='Integer tif file generated by patch_metrics.py')
    parser.add_argument('--wind_data', help='NetCDF with eastward and northward wind speed generated by barra_daily.py')
    parser.add_argument('--height_tif', help='Integer tif file generated by canopy_height.py')
    parser.add_argument('--outdir', default='.', help='Output directory for saving results (default: current directory)')
    parser.add_argument('--stub', default='TEST', help='Prefix for output filenames (default: TEST)')
    parser.add_argument('--wind_method', default='WINDWARD', help="Either 'WINDWARD', 'MOST_COMMON', 'MAX', 'HAPPENED', 'ANY', or 'MULTI_LAYER' (default: WINDWARD)")
    parser.add_argument('--distance_threshold', default=20, type=int, help='Distance from trees that counts as sheltered (default: 20)')
    parser.add_argument('--wind_threshold', default=20, type=int, help='Wind speed threshold in km/h (default: 20)')
    parser.add_argument('--density_threshold', default=5, type=int, help='Percentage tree cover within distance_threshold that counts as sheltered (default: 5)')
    parser.add_argument('--crop_pixels', default=None, type=int, help='Number of pixels to crop from each edge of the output (default: no cropping)')
    parser.add_argument('--no-savetif', dest='savetif', action="store_false", default=True, help="Disable saving GeoTIFF output (default: enabled)")
    parser.add_argument('--no-plot', dest='plot', action="store_false", default=True, help="Disable PNG visualisation (default: enabled)")
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug output')

    return parser


if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    ds = shelter_categories(
        args.linear_data,
        wind_data=args.wind_data,
        height_tif=args.height_tif,
        outdir=args.outdir,
        stub=args.stub,
        wind_method=args.wind_method,
        wind_threshold=args.wind_threshold,
        distance_threshold=args.distance_threshold,
        density_threshold=args.density_threshold,
        savetif=args.savetif,
        plot=args.plot,
        crop_pixels=args.crop_pixels,
    )
