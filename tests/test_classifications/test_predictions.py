import os
import pickle
import joblib

from shelterbelts.classifications.predictions import tif_prediction_ds, _load_keras_model
from shelterbelts.utils.filepaths import sentinel_sample, get_pretrained_nn, get_pretrained_scaler


def _load_ds():
    with open(sentinel_sample, 'rb') as f:
        return pickle.load(f)


def test_tif_prediction_ds_binary():
    ds = _load_ds()
    model = _load_keras_model(get_pretrained_nn())
    scaler = joblib.load(get_pretrained_scaler())
    da = tif_prediction_ds(ds, outdir='outdir', stub='test_pred', model=model, scaler=scaler, savetif=True)
    assert da.dtype.name == 'uint8'
    assert set(da.values.flatten().tolist()).issubset({0, 1})
    assert os.path.exists('outdir/test_pred_predicted.tif')


def test_tif_prediction_ds_confidence():
    ds = _load_ds()
    model = _load_keras_model(get_pretrained_nn())
    scaler = joblib.load(get_pretrained_scaler())
    da = tif_prediction_ds(ds, outdir='outdir', stub='test_conf', model=model, scaler=scaler, confidence=True, savetif=False)
    assert da.dtype.name == 'uint8'
    assert da.values.min() >= 0
    assert da.values.max() <= 100


def test_tif_prediction_ds_multi_model():
    ds = _load_ds()
    da = tif_prediction_ds(ds, outdir='outdir', stub='test_multi', weighted_average=True, savetif=False)
    assert da.dtype.name == 'uint8'
    assert set(da.values.flatten().tolist()).issubset({0, 1})
