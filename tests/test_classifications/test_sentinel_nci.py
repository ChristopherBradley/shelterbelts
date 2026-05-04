import os

import pytest

from shelterbelts.utils.filepaths import IS_GADI
from shelterbelts.classifications.sentinel_nci import (
    define_query_range,
    load_and_process_data,
    download_ds2_bbox,
)

BBOX = (149.0, -35.5, 149.1, -35.4)
START = '2020-01-01'
END   = '2020-04-01'

pytestmark = pytest.mark.skipif(not IS_GADI, reason="NCI/Gadi datacube not available")


def test_load_and_process_data_shape():
    import datacube
    lat_range = (BBOX[1], BBOX[3])
    lon_range = (BBOX[0], BBOX[2])
    query = define_query_range(lat_range, lon_range, (START, END))
    dc = datacube.Datacube(app='test_sentinel_nci')
    ds = load_and_process_data(dc, query)
    bands = [
        'nbart_blue', 'nbart_green', 'nbart_red',
        'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3',
        'nbart_nir_1', 'nbart_nir_2',
        'nbart_swir_2', 'nbart_swir_3',
    ]
    assert set(ds.data_vars) == set(bands)
    assert set(ds.dims) == {'time', 'x', 'y'}
    assert ds.dims['time'] >= 1


def test_download_ds2_bbox_saves_pickle():
    download_ds2_bbox(BBOX, START, END, outdir='outdir', stub='TEST_nci', save=True)
    assert os.path.exists('outdir/TEST_nci_ds2_2020.pkl')
