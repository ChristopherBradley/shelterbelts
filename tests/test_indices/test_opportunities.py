import numpy as np
import xarray as xr

from shelterbelts.indices.opportunities import opportunities_da


def _make_da(data):
    return xr.DataArray(np.array(data, dtype=np.uint8), dims=["y", "x"])


def test_opportunities_da_basic():
    """Basic test for opportunities_da using synthetic inputs."""
    trees = _make_da(np.zeros((7, 7), dtype=np.uint8))
    worldcover = _make_da(np.full((7, 7), 30, dtype=np.uint8))

    gullies = _make_da(np.zeros((7, 7), dtype=np.uint8))
    roads = _make_da(np.zeros((7, 7), dtype=np.uint8))
    ridges = _make_da(np.zeros((7, 7), dtype=np.uint8))
    contours = _make_da(np.zeros((7, 7), dtype=np.uint8))

    gullies.values[1, 1] = 1
    roads.values[1, 5] = 1
    ridges.values[5, 1] = 1
    contours.values[5, 5] = 1

    ds = opportunities_da(
        trees,
        roads,
        gullies,
        ridges,
        contours,
        worldcover,
        outdir="outdir",
        stub="test",
        width=1,
        savetif=False,
        plot=False,
        crop_pixels=0,
    )

    assert set(ds.data_vars) == {"woody_veg", "opportunities"}

    # Check specific locations match expected categories
    assert ds["opportunities"].values[1, 1] == 5  # gullies
    assert ds["opportunities"].values[1, 5] == 7  # roads
    assert ds["opportunities"].values[5, 1] == 6  # ridges
    assert ds["opportunities"].values[5, 5] == 8  # contours


def test_opportunities_da_no_ridges():
    """When ridges are None, ridge opportunities should be absent."""
    trees = _make_da(np.zeros((5, 5), dtype=np.uint8))
    worldcover = _make_da(np.full((5, 5), 30, dtype=np.uint8))

    gullies = _make_da(np.zeros((5, 5), dtype=np.uint8))
    roads = _make_da(np.zeros((5, 5), dtype=np.uint8))
    contours = _make_da(np.zeros((5, 5), dtype=np.uint8))

    gullies.values[2, 1] = 1
    roads.values[1, 3] = 1
    contours.values[3, 3] = 1

    ds = opportunities_da(
        trees,
        roads,
        gullies,
        None,
        contours,
        worldcover,
        outdir="outdir",
        stub="test_no_ridges",
        width=1,
        savetif=False,
        plot=False,
        crop_pixels=0,
    )

    assert set(ds.data_vars) == {"woody_veg", "opportunities"}
    assert 6 not in np.unique(ds["opportunities"].values)
