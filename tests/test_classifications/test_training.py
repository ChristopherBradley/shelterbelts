import os

from shelterbelts.classifications.neural_network import train_neural_network
from shelterbelts.classifications.random_forest import random_forest
from shelterbelts.utils.filepaths import training_csv_sample


def test_random_forest_output():
    df = random_forest(training_csv_sample, outdir='outdir', stub='demo_rf')
    assert os.path.exists('outdir/demo_rf_random_forest.pkl')
    assert os.path.exists('outdir/scaler_demo_rf.pkl')
    assert set(df.columns) >= {'precision', 'recall', 'f1-score', 'accuracy'}
    assert set(df['tree_class']) == {0.0, 1.0}


def test_neural_network_output():
    df = train_neural_network(training_csv_sample, outdir='outdir', stub='demo_nn', epochs=5)
    assert os.path.exists('outdir/nn_demo_nn.keras')
    assert os.path.exists('outdir/scaler_demo_nn.pkl')
    assert set(df.columns) >= {'precision', 'recall', 'f1-score', 'accuracy'}
    assert set(df['tree_class']) == {0.0, 1.0}


def test_accuracy_comparison():
    """Both models return the same DataFrame shape so they can be compared directly."""
    import pandas as pd
    df_rf = random_forest(training_csv_sample, outdir='outdir', stub='demo_rf')
    df_nn = train_neural_network(training_csv_sample, outdir='outdir', stub='demo_nn', epochs=5)
    df_rf['model'] = 'random_forest'
    df_nn['model'] = 'neural_network'
    combined = pd.concat([df_rf, df_nn])
    assert len(combined) == 4  # 2 classes × 2 models
    assert set(combined['model']) == {'random_forest', 'neural_network'}
