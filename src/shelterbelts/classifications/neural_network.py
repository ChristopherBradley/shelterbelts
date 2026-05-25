import os
import argparse
import logging

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Quiet tensorflow import-time logging.
logging.getLogger("numexpr.utils").setLevel(logging.ERROR)

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


def my_train_test_split(df, stratification_columns=[], train_frac=0.7, random_state=0):
    """Stratified train/test split that downsamples each class to the minimum-class count."""
    if len(stratification_columns) == 0:
        # Using an empty list for stratification_columns means no stratification
        df_train, df_test = train_test_split(
            df,
            train_size=train_frac,
            random_state=random_state,
            shuffle=True
        )
        stratification_columns = 'tree_cover'

    else:
        samples_per_class = min(df[stratification_columns].value_counts())
        df_stratified = (
            df
            .groupby(stratification_columns, group_keys=False)
            .sample(n=samples_per_class, random_state=random_state)
        )
        train_list = []
        test_list = []
        for _, group in df_stratified.groupby(stratification_columns):
            train, test = train_test_split(
                group,
                train_size=train_frac,
                random_state=random_state,
                shuffle=True
            )
            train_list.append(train)
            test_list.append(test)
        df_train = pd.concat(train_list).reset_index(drop=True)
        df_test = pd.concat(test_list).reset_index(drop=True)

    print(f"Original number of samples: {len(df)}")
    print(df[stratification_columns].value_counts().sort_index())
    
    print(f"\n\nNumber of training samples: {len(df_train)}")
    print(df_train[stratification_columns].value_counts().sort_index())
    
    print(f"\n\nNumber of testing samples: {len(df_test)}")
    print(df_test[stratification_columns].value_counts().sort_index())

    # Some full validation tiles should also be preserved during merge_inputs_outputs before training.

    return df_train, df_test


def inputs_outputs_split(df_train, df_test, outdir, stub, non_input_variables, output_column='tree_cover'):
    """Standard-scale the input columns, one-hot encode output_column, and save the scaler."""
    # Normalise the input features
    scaler = StandardScaler()     # Should maybe also remove outliers before scaling.
    X_train = scaler.fit_transform(df_train.drop(columns=non_input_variables, errors='ignore'))
    X_test = scaler.transform(df_test.drop(columns=non_input_variables, errors='ignore'))

    # Save the StandardScaler
    filename_scaler = os.path.join(outdir, f'scaler_{stub}.pkl')
    joblib.dump(scaler, filename_scaler)  
    print("Saved", filename_scaler)

    # One-hot encode the output features
    y_train = keras.utils.to_categorical(df_train[output_column], 2)
    y_test = keras.utils.to_categorical(df_test[output_column], 2)

    return X_train, X_test, y_train, y_test, scaler

    
def train_model(X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size, outdir='TEST', stub='.'):
    """Train a 3-layer dense NN with early stopping, save the keras file, and plot accuracy/loss curves."""
    dropout=0.1
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu'),    
        keras.layers.Dropout(dropout), 

        keras.layers.Dense(128, activation='relu'),    
        keras.layers.Dropout(dropout), 
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(dropout), 
        
        keras.layers.Dense(2, activation='softmax')
    ])
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=30,           # For some reason the BWh training still seemed to be impatient after ~10 epochs
        restore_best_weights=True 
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # Tried out a few different optimizers and this worked best for my use case

    model.compile(optimizer=optimizer, loss = 'CategoricalCrossentropy', metrics = ['CategoricalAccuracy'])        

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        callbacks=[early_stopping], validation_data=(X_test, y_test), verbose=2)

    # Save the model
    filename = os.path.join(outdir, f'nn_{stub}.keras')
    model.save(filename)
    print("Saved", filename)
    
    # Accuracy and Loss Plots
    history_df = pd.DataFrame.from_dict(history.history)

    # Different versions of keras have different naming conventions for the accuracy and loss
    def norm(k: str) -> str:
        return k.lower().replace("_", "")
    desired = {
        "categoricalaccuracy": "Training Accuracy",
        "valcategoricalaccuracy": "Testing Accuracy",
        "loss": "Training Loss",
        "valloss": "Testing Loss",
    }
    rename_map = {col: desired[norm(col)] for col in history_df.columns if norm(col) in desired}
    
    history_df = history_df.rename(columns = rename_map)
    
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    
    sns.lineplot(ax=axes[0], data=history_df[['Training Accuracy', 'Testing Accuracy']])
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    
    sns.lineplot(ax=axes[1], data=history_df[['Training Loss', 'Testing Loss']])
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    
    plt.tight_layout()
    filename = os.path.join(outdir, f'{stub}_training_plots.png')
    plt.savefig(filename)
    print("Saved", filename)

    return model


def class_accuracies_stratified(df_test, model, scaler, outdir, stub, non_input_variables):
    """Per-Koppen-class plus overall precision/recall/accuracy; writes {stub}_accuracy.csv."""
    X_test = scaler.transform(df_test.drop(columns=non_input_variables))
    y_pred_percent = model.predict(X_test)
    y_pred = [percent.argmax() for percent in y_pred_percent]
    
    results = df_test[['tree_cover', 'koppen_class']].copy()
    results['y_pred'] = y_pred
    rf_rows = []
    for koppen_class in results['koppen_class'].unique():
        subset = results[results['koppen_class'] == koppen_class]
        report = classification_report(subset['tree_cover'], subset['y_pred'], output_dict=True, zero_division=0)
        accuracy = accuracy_score(subset['tree_cover'], subset['y_pred'])
        for tree_class in [0.0, 1.0]:
            if str(tree_class) in report:
                rf_rows.append({
                    'koppen_class': koppen_class,
                    'tree_class': tree_class,
                    'precision': report[str(tree_class)]['precision'],
                    'recall': report[str(tree_class)]['recall'],
                    'f1-score': report[str(tree_class)]['f1-score'],
                    'accuracy': accuracy,
                    'support': report[str(tree_class)]['support'],
                })

    # Overall metrics 
    rf_metrics_table = pd.DataFrame(rf_rows)
    overall_report = classification_report(results['tree_cover'], results['y_pred'], output_dict=True, zero_division=0)
    overall_accuracy = accuracy_score(results['tree_cover'], results['y_pred'])
    for tree_class in [0.0, 1.0]:
        if str(tree_class) in overall_report:
            rf_metrics_table.loc[len(rf_metrics_table)] = {
                'koppen_class': 'overall',
                'tree_class': tree_class,
                'precision': overall_report[str(tree_class)]['precision'],
                'recall': overall_report[str(tree_class)]['recall'],
                'f1-score': overall_report[str(tree_class)]['f1-score'],
                'accuracy': overall_accuracy,
                'support': overall_report[str(tree_class)]['support'],
            }
    filename = os.path.join(outdir, f'{stub}_metrics.csv')
    rf_metrics_table.to_csv(filename)
    print("Saved", filename)

    # Reshape to match metrics from Stewart et al. 2025
    koppen_classes = rf_metrics_table['koppen_class'].unique()
    dfs = []
    for koppen_class in koppen_classes:
        df = rf_metrics_table[(rf_metrics_table['koppen_class']==koppen_class) & rf_metrics_table['tree_class']==1]
        df = df.drop(columns=['tree_class'])
        df = df.rename(columns={'support': 'test_samples', 'f1-score':'specificity'})
        df['specificity'] = rf_metrics_table.loc[(rf_metrics_table['koppen_class']==koppen_class) & (rf_metrics_table['tree_class']==0), 'recall'].iloc[0]
        df['test_samples'] = rf_metrics_table.loc[(rf_metrics_table['koppen_class']==koppen_class), 'support'].sum()
        dfs.append(df)
    df_combined = pd.concat(dfs)
    
    metrics = ['precision','recall','specificity','accuracy']
    for metric in metrics:
        df_combined[metric] = (df_combined[metric] * 100).round().astype(int)
    df_combined['test_samples'] = df_combined['test_samples'].astype(int)
    
    filename = os.path.join(outdir, f'{stub}_accuracy.csv')
    df_combined = df_combined.sort_values(by='koppen_class')
    df_combined.to_csv(filename, index=False)
    print("Saved", filename)

    return rf_metrics_table
    
def class_accuracies_overall(df_test, model, scaler, outdir, stub, non_input_variables, output_column):
    """Overall precision/recall/accuracy on the testing data."""
    X_test = scaler.transform(df_test.drop(columns=non_input_variables, errors='ignore'))
    y_pred_percent = model.predict(X_test)
    y_pred = [percent.argmax() for percent in y_pred_percent]
    results = df_test[[output_column]].copy()
    results['y_pred'] = y_pred
    overall_report = classification_report(results[output_column], results['y_pred'], output_dict=True, zero_division=0)
    overall_accuracy = accuracy_score(results[output_column], results['y_pred'])
    rf_rows = []
    tree_classes = list(df_test[output_column].unique())
    for tree_class in tree_classes:
        if str(tree_class) in overall_report:
            rf_rows.append( {
                'tree_class': tree_class,
                'precision': overall_report[str(tree_class)]['precision'],
                'recall': overall_report[str(tree_class)]['recall'],
                'f1-score': overall_report[str(tree_class)]['f1-score'],
                'accuracy': overall_accuracy,
                'support': overall_report[str(tree_class)]['support'],
            })
    rf_metrics_table = pd.DataFrame(rf_rows)
    filename = os.path.join(outdir, f'{stub}_metrics.csv')
    rf_metrics_table.to_csv(filename)
    print("Saved", filename)
    print(rf_metrics_table)  # More convenient to see this directly in the pbs output than to have to open the csv files
    
    return rf_metrics_table


def train_neural_network(training_file, outdir=".", stub="TEST", output_column='tree_cover', drop_columns=['x', 'y', 'tile_id'], learning_rate=0.001, epochs=50, batch_size=32, random_state=1, stratification_columns=['tree_cover'], train_frac=0.7, limit=None):
    """
    Train and evaluate a dense neural network to classify pixels as tree or non-tree.

    Parameters
    ----------
    training_file : str
        Path to a .csv or .feather training dataset. Must contain the
        output_column plus the per-pixel feature columns produced by
        merge_inputs_outputs.
    outdir : str, optional
        Output directory for saving results.
    stub : str, optional
        Prefix for every output filename.
    output_column : str, optional
        Name of the binary label column.
    drop_columns : list of str, optional
        Non-feature columns to drop before training.
    learning_rate : float, optional
        Hyperparameter for tuning.
    epochs : int, optional
        Maximum epochs (however, early stopping might terminate sooner than this).
    batch_size : int, optional
        Hyperparameter for tuning.
    random_state : int, optional
        Seed for the train/test split and subsampling.
    stratification_columns : list of str, optional
        Columns to stratify the train/test split on. An empty list disables stratification.
    train_frac : float, optional
        Fraction of stratified samples used for training.
    limit : int, optional
        Randomly subsample the training file to this many rows before the
        split. By default this uses all rows.

    Returns
    -------
    pandas.DataFrame
        Precision/recall/f1/accuracy per class on the test data.

    Notes
    -----
    Writes to outdir:

    - nn_{stub}.keras 
    - scaler_{stub}.pkl
    - {stub}_metrics.csv 
    - {stub}_training_plots.png 

    Examples
    --------
    >>> from shelterbelts.utils.filepaths import training_csv_sample
    >>> df = train_neural_network(training_csv_sample, outdir='/tmp',
    ...                           stub='example', epochs=1, limit=500)  # doctest: +SKIP
    """
    if training_file.endswith('.feather'):
        df = pd.read_feather(training_file)
    else:
        df = pd.read_csv(training_file)

    df = df[df.notna().all(axis=1)]

    if limit:
        df = df.sample(limit, random_state=random_state)

    df_train, df_test = my_train_test_split(df, stratification_columns, train_frac, random_state)

    non_input_columns = [output_column] + drop_columns
    X_train, X_test, y_train, y_test, scaler = inputs_outputs_split(df_train, df_test, outdir, stub, non_input_columns, output_column)

    model = train_model(X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size, outdir, stub)

    if 'koppen_class' in df_test.columns:
        df_metrics = class_accuracies_stratified(df_test, model, scaler, outdir, stub, non_input_columns)
    else:
        df_metrics = class_accuracies_overall(df_test, model, scaler, outdir, stub, non_input_columns, output_column)

    print(df_metrics)
    return df_metrics


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('training_file', help='Either a .feather or .csv file, generated by merge_inputs_outputs.csv')
    parser.add_argument('--outdir', default='.', help='The directory for the outputs (default: current directory)')
    parser.add_argument('--stub', default='TEST_NN', help='Prefix of the outputs (default: TEST_NN)')
    parser.add_argument('--output_column', default='tree_cover', help='Column with the output data (default: tree_cover)')
    parser.add_argument('--drop_columns', nargs='+', default=['x', 'y', 'tile_id'], help='Columns to drop (default: x y tile_id)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate hyperparameter (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 32)')
    parser.add_argument('--random_state', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--stratification_columns', nargs='+', default=['tree_cover'], help='Columns to stratify samples on (default: tree_cover)')
    parser.add_argument('--train_frac', type=float, default=0.7, help='Fraction of samples to use for training (default: 0.7)')
    parser.add_argument('--limit', type=int, default=None, help='Number of rows to use when training (default: all)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    train_neural_network(
        args.training_file,
        outdir=args.outdir,
        stub=args.stub,
        output_column=args.output_column,
        drop_columns=args.drop_columns,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        stratification_columns=args.stratification_columns,
        train_frac=args.train_frac,
        limit=args.limit,
    )
