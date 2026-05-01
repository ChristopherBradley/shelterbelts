# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Predicting Unknown Data
#

# %%
import pickle
import joblib

from shelterbelts.classifications.predictions import tif_prediction_ds, _load_keras_model
from shelterbelts.utils.filepaths import sentinel_sample, get_pretrained_nn, get_pretrained_scaler

# %%
# Load some example sentinel imagery
with open(sentinel_sample, 'rb') as f:
    ds = pickle.load(f)

ds

# %%
# Load the pre-trained model
model = _load_keras_model(get_pretrained_nn())
scaler = joblib.load(get_pretrained_scaler())

# %% [markdown]
# ## 1. Binary tree classification

# %%
# %%time
da = tif_prediction_ds(ds, model=model, scaler=scaler)
da.plot(cmap='Greens')

# %% [markdown]
# ## 2. Tree likelihood predictions

# %%
# %%time
da_conf = tif_prediction_ds(ds, model=model, scaler=scaler, confidence=True)
da_conf.plot(cmap='Greens')

# %% [markdown]
# ## 3. Weighted average of multiple models (1 per koppen region)
#

# %%
# %%time
da_multi = tif_prediction_ds(ds, outdir='outdir', stub='demo_multi', weighted_average=True)
da_multi.plot(cmap='Greens')
