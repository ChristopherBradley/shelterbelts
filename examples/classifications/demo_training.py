# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training tree-cover classifiers
#
# ``random_forest`` and ``neural_network`` (multi-layer perceptron) are two
# classifiers that predict tree vs non-tree pixels from temporally-aggregated
# Sentinel-2 features. This notebook trains both on the same sample CSV and
# compares their testing accuracy.

# %%
from shelterbelts.classifications.random_forest import random_forest
from shelterbelts.classifications.neural_network import train_neural_network
from shelterbelts.utils.filepaths import training_csv_sample

# %% [markdown]
# ## Random forest

# %%
# %%time
df_rf = random_forest(
    training_csv_sample,
    outdir='outdir',
    stub='demo_rf'
)
df_rf

# %% [markdown]
# ## Neural network (MLP)

# %%
# %%time
df_nn = train_neural_network(
    training_csv_sample,
    outdir='outdir',
    stub='demo_nn',
    epochs=5
)
df_nn

# %% [markdown]
# ## Compare accuracy

# %%
import pandas as pd

df_rf['model'] = 'random_forest'
df_nn['model'] = 'neural_network'
pd.concat([df_rf, df_nn]).set_index(['model', 'tree_class'])[['precision', 'recall', 'f1-score', 'accuracy']]
