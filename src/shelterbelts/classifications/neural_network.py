# +
# Train a neural network to compare with random forest predictions

# +
# # !pip install pyarrow # For loading .feather files

# +
# %%time
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

# Takes 1 min to load all the libraries on my mac


def attach_koppen_classes(df):
    """Attach the koppen class of each tile to the relevant rows"""
    # This should really all happen in merge_inputs_outputs
    df = df[(df['tree_cover'] == 0) | (df['tree_cover'] == 1)]     # Drop the 174 rows where tree cover values = 2
    df = df[df.notna().all(axis=1)]     # Drop the two rows with NaN values

    # Add the bioregion to the training/testing data
    gdf = gpd.read_file(filename_centroids)

    gdf['tile_id'] = ["_".join(filename.split('/')[-1].split('_')[:2]) for filename in gdf['filename']]
    gdf = gdf.rename(columns={'Full Name':'koppen_class'})
    df = df.merge(gdf[['tile_id', 'koppen_class']])

    # Should also use the y, x, tile_id to get a coord in the Australia EPSG:7844

    return df

def my_train_test_split(df, stratification_columns, train_frac, random_state):
    """Stratified train test split"""
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

    # Some full validation tiles should also have be preserved during merge_inputs_outputs

    return df_train, df_test

def inputs_outputs_split(df_train, df_test, outdir, stub, non_input_variables):
    """Final prepping of data for neural network training"""
    # Normalise the input features
    scaler = StandardScaler()     # I should probably remove outliers before scaling.
    X_train = scaler.fit_transform(df_train.drop(columns=non_input_variables))
    X_test = scaler.transform(df_test.drop(columns=non_input_variables))

    # Save the StandardScaler
    filename_scaler = os.path.join(outdir, f'scaler_{stub}.pkl')
    joblib.dump(scaler, filename_scaler)  
    print("Saved", filename_scaler)

    # One-hot encode the output features
    y_train = keras.utils.to_categorical(df_train['tree_cover'], 2)
    y_test = keras.utils.to_categorical(df_test['tree_cover'], 2)

    return X_train, X_test, y_train, y_test, scaler

    
def train_model(X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size):
    """Train a neural network and save the resulting model and training plots"""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),    
        keras.layers.Dropout(0.3), 
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3), 
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5,           # Stop training if val_loss doesn't improve n epochs
        restore_best_weights=True 
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss = 'CategoricalCrossentropy', metrics = ['CategoricalAccuracy'])        

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                        callbacks=[early_stopping], validation_data=(X_test, y_test), verbose=2)

    # Save the model
    filename = os.path.join(outdir, f'nn_{stub}.keras')
    model.save(filename)
    print("Saved", filename)
    
    # Accuracy and Loss Plots
    history_df = pd.DataFrame.from_dict(history.history)
    history_mapping = {'CategoricalAccuracy': 'Training Accuracy', 'val_CategoricalAccuracy': 'Testing Accuracy', 'loss': "Training Loss", 'val_loss': "Testing Loss"}
    history_df = history_df.rename(columns = history_mapping)
    
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

    return model


def class_accuracies(df_test, model, scaler, outdir, stub):
    """Calculate overall and per class accuracy metrics"""
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
    filename = os.path.join(outdir, f'metrics_{stub}.csv')
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
    
    filename = os.path.join(outdir, f'accuracy_{stub}.csv')
    df_combined = df_combined.sort_values(by='koppen_class')
    df_combined.to_csv(filename, index=False)
    print("Saved", filename)

    return rf_metrics_table
    

def neural_network(outdir, stub, learning_rate=0.001, epochs=50, batch_size=32, random_state=1, stratification_columns=[], train_frac = 0.7, limit=None):
    """
    Create and evaluate a neural network to predict tree vs no tree classifications

    Parameters
    ----------
        outdir: 
        stub: 
        stratify: Whether to normalise the number of samples from each class (reduces the overall number of training samples)
        limit: Number of rows to use when training the model

    Returns
    -------
        df_accuracy: precision, recall, specificity, sensitivity, accuracy for 0's and 1's - and grouped by category

    Downloads
    ---------
        accuracy_metrics: The df_accuracy as a csv
        model.keras: The machine learning model for running on new input data
        scaler.pkl: The standard scaler used to normalise the input data
        training.png: accuracy and loss plots over each epoch
    """
    feather_filename = os.path.join(outdir, f"tree_cover_preprocessed_{stub}.feather")
    df = pd.read_feather(feather_filename)

    if limit:
        df = df.sample(limit, random_state=random_state)

    filename_centroids = "/Users/christopherbradley/Documents/PHD/Data/Nick_outlines/centroids_named.gpkg"
    df = attach_koppen_classes(df)

    df_train, df_test = my_train_test_split(df, stratification_columns, train_frac, random_state)
    
    non_input_columns = ['tree_cover', 'y', 'x', 'tile_id', 'koppen_class']  # Might be better to invert this so my_train_test_split takes the input_columns instead of non_input_columns
    X_train, X_test, y_train, y_test, scaler = inputs_outputs_split(df_train, df_test, outdir, stub, non_input_variables)
    
    model = train_model(X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size)

    df_metrics = class_accuracies(df_test, model, scaler, outdir, stub)



# -

# %%time
if __name__ == '__main__':

    outdir = "/Users/christopherbradley/Documents/PHD/Data/Nick_models"
    stub = "kernel4"
    neural_network(outdir, stub, stratification_columns=['tree_cover'], limit=10000)

