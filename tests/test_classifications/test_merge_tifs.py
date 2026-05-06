import os
import numpy as np
import rasterio

from shelterbelts.classifications.merge_tifs import merge_tifs
from shelterbelts.utils.filepaths import quartered_tifs_dir


def test_merge_tifs_basic():
    """Four corner tiles merge into a single raster covering the full extent."""
    da = merge_tifs(quartered_tifs_dir, tmpdir='tmpdir', suffix='.tif', dont_reproject=True)
    outpath = os.path.join(os.path.dirname(quartered_tifs_dir), 'quartered_binary_tifs_merged.tif')
    assert os.path.exists(outpath)
    assert da.shape[0] > 0 and da.shape[1] > 0


def test_merge_tifs_colormap():
    """Merged output tif carries the colormap from the input tiles."""
    merge_tifs(quartered_tifs_dir, tmpdir='tmpdir', suffix='.tif', dont_reproject=True)
    outpath = os.path.join(os.path.dirname(quartered_tifs_dir), 'quartered_binary_tifs_merged.tif')
    with rasterio.open(outpath) as src:
        cmap = src.colormap(1)
    assert cmap[12][:3] == (8, 79, 0)  # Linear-categories green for value 12 (Patch Core)


def test_merge_tifs_dedup():
    """With dedup=True the older binary TL tile (2019) is discarded and the
    2020 linear-categories tile is kept, so merged values exceed 1."""
    da = merge_tifs(quartered_tifs_dir, tmpdir='tmpdir', suffix='.tif',
                    dont_reproject=True, dedup=True)
    outpath = os.path.join(os.path.dirname(quartered_tifs_dir), 'quartered_binary_tifs_merged.tif')
    assert os.path.exists(outpath)
    assert max(np.unique(da.values)) > 1  # The 2020 linear-categories tile contains values > 1, confirming it was kept over the 2019 tile.
