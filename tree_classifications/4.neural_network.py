# +
# Train a neural network to compare with random forest predictions
# -

# %%time
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# +
# %%time
# This is getting up towards 1GB so I should really save as a feather file now. Doesn't need to be readable because I can't open a csv of 1GB without things crashing anyway. 
outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"
filename = os.path.join(outlines_dir, f"tree_cover_preprocessed2.csv")
df = pd.read_csv(filename, index_col=False)

# Takes 10 secs to load a csv with 1 million rows

# +
# Drop the 174 rows where tree cover values = 2
df = df[(df['tree_cover'] == 0) | (df['tree_cover'] == 1)]

# Drop the two rows with NaN values
df = df[df.notna().all(axis=1)]
# -

len(df)

# +
# %%time
# Read in the bioregions
filename_centroids = os.path.join(outlines_dir, f"centroids_named.gpkg")
gdf = gpd.read_file(filename_centroids)

# Add the bioregion to the training/testing data
gdf['tile_id'] = ["_".join(filename.split('/')[-1].split('_')[:2]) for filename in gdf['filename']]
gdf = gdf.rename(columns={'Full Name':'koppen_class'})
df = df.merge(gdf[['tile_id', 'koppen_class']])
# -

# Randomise the samples
random_state = 1
df_sample_full = df.sample(n=len(df), random_state=random_state)  # randomising everything so I can later use a larger random testing dataset while being sure I don't reuse training data

# Take an equal number of samples from each koppen and tree cover class
samples_per_class = min(df_sample_full[['koppen_class', 'tree_cover']].value_counts())
df_stratified = (
    df_sample_full
    .groupby(['koppen_class', 'tree_cover'], group_keys=False)
    .sample(n=samples_per_class, random_state=random_state)
)
df_stratified[['koppen_class', 'tree_cover']].value_counts()

# +
# Create training and testing data with equally distributed classes
train_list = []
test_list = []

train_frac = 0.7
for _, group in df_stratified.groupby(['koppen_class', 'tree_cover']):
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
# -

df_test[['koppen_class', 'tree_cover']].value_counts()

# Trying out training on all the data instead.
df_stratified = df_sample_full
sample_size = int(len(df_stratified) * 0.7)
df_train = df_stratified[:sample_size]
df_test = df_stratified[sample_size:]

# +
# Normalise the input features (should probs do this before creating the .feather file)
X = df_train.drop(columns=['tree_cover', 'y', 'x', 'tile_id', 'koppen_class']) # input variables
X = StandardScaler().fit_transform(X)

y = df_train['tree_cover']  # target variable
y_categorical = keras.utils.to_categorical(y, 2)

# Doing the train test split earlier on so I can play around with stratification
X_train = X
y_train = y_categorical

# +
# Should clean up this train test split
X = df_test.drop(columns=['tree_cover', 'y', 'x', 'tile_id', 'koppen_class']) # input variables
X_test = StandardScaler().fit_transform(X)

y = df_test['tree_cover']  # target variable
y_test = keras.utils.to_categorical(y, 2)
# -

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5,           # Stop training if val_loss doesn't improve n epochs
    restore_best_weights=True 
)

# +
# %%time
# Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),    
    keras.layers.Dropout(0.3), 
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3), 
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss = 'CategoricalCrossentropy', metrics = 'CategoricalAccuracy')        
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    callbacks=[early_stopping], validation_data=(X_test, y_test))

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['categorical_accuracy', 'val_categorical_accuracy']])
plt.show()

# +
# 89% accuracy and 1 min 21 secs
# 89% accuracy and 14 mins with small compute, all 1 million training samples

# +
# Evaluate the accuracy for each bioregion using unused data (a larger sample than the designated test data, but also not used in the training data)

# Double check we aren't reusing any of the training data for testing each bioregion
print("Number of samples taken from training data:", len(df_train[df_train.index.isin(df_test.index)]))

# Predict all 100k of these bioregion testing datapoints
X = df_test.drop(columns=['tree_cover','y', 'x', 'tile_id', 'koppen_class']) 
X_normalized = StandardScaler().fit_transform(X) # Is this scaling consistent across different datasets?

y_test = df_test[['tree_cover', 'koppen_class']]
y_pred_percent = model.predict(X_normalized)
y_pred = [percent.argmax() for percent in y_pred_percent]
# print(classification_report(y_test['tree_cover'], y_pred))

# Join predictions with true values and koppen_class
results = df_test[['tree_cover', 'koppen_class']].copy()
results['y_pred'] = y_pred

# Collect rows for the summary table
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

# Create the summary table
rf_metrics_table = pd.DataFrame(rf_rows)

# Compute overall metrics (across all koppen_class values)
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

rf_metrics_table


# +
# Reshape to match metrics from Stewarts et al. 2025
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

df_combined
# -


