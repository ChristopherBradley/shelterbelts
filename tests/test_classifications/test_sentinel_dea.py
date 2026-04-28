import os

import pytest
import requests

from shelterbelts.classifications.sentinel_dea import search_stac, load_and_process_data, download_ds2_bbox, BANDS, STAC_URL

BBOX = (149.0, -35.5, 149.1, -35.4)
START = '2020-01-01'
END   = '2020-04-01'


def _dea_available():
    try:
        return requests.get(STAC_URL, timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _dea_available(), reason="DEA STAC endpoint unavailable")


def test_search_stac_returns_items():
    items = search_stac(BBOX, START, END)
    assert len(items) >= 1


def test_load_and_process_data_shape():
    items = search_stac(BBOX, START, END)
    ds = load_and_process_data(items, BBOX)
    assert set(ds.data_vars) == set(BANDS)
    assert set(ds.dims) == {'time', 'x', 'y'}
    assert ds.dims['time'] >= 1


def test_download_ds2_bbox_saves_pickle():
    download_ds2_bbox(BBOX, START, END, outdir='outdir', stub='TEST_dea', save=True)
    assert os.path.exists('outdir/TEST_dea_ds2_2020.pkl')
