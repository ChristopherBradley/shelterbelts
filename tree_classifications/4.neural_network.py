# +
# Train a neural network to compare with random forest predictions
# -

import os
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# %%time
outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"
filename = os.path.join(outlines_dir, f"tree_cover_preprocessed.csv")
df = pd.read_csv(filename, index_col=False)

# +
# Drop the 174 rows where tree cover values = 2
df = df[(df['tree_cover'] == 0) | (df['tree_cover'] == 1)]

# Drop the two rows with NaN values
df = df[df.notna().all(axis=1)]

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

# sample_size = 100000
random_state = 0
df_sample_full = df.sample(n=len(df), random_state=random_state)  # randomising everything so I can later use a larger random testing dataset while being sure I don't reuse training data
# df_sample = df_sample_full[:sample_size]

# Make the training dataset have an equal number of tree vs no tree classes, so it doesn't over predict no trees. 
samples_per_class = len(df_sample_full[df_sample_full['tree_cover'] == 1])
df_stratified = (
    df_sample_full
    .groupby('tree_cover', group_keys=False)
    .sample(n=samples_per_class, random_state=random_state)
)
df_stratified = df_stratified.sample(len(df_stratified), random_state=random_state)

df_stratified = df_stratified[df_stratified['koppen_class'] == 'Tropical savanna QLD']

# sample_size = int(len(df_single_class) * 0.7)
sample_size = int(len(df_stratified) * 0.7)

df_sample = df_stratified[:sample_size]

print(len(df_sample))
df_sample['koppen_class'].value_counts()

df_sample.shape

# +
# Normalise the input features (should probs do this before creating the .feather file)
X = df_sample.drop(columns=['tree_cover', 'y', 'x', 'tile_id', 'koppen_class']) # input variables
X = StandardScaler().fit_transform(X)

y = df_sample['tree_cover']  # target variable
y_categorical = keras.utils.to_categorical(y, 2)

# Split the data into training and testing sets (70% train, 30% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=random_state)

# Doing the train test split earlier on so I can play around with stratification
X_train = X
y_train = y_categorical

# +
# Should clean up this train test split
df_test = df_stratified[sample_size:]
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

# +
# Evaluate the accuracy for each bioregion using unused data (a larger sample than the designated test data, but also not used in the training data)
# df_bioregion_test = df_sample_full[600000:700000]
df_bioregion_test = df_test

# Double check we aren't reusing any of the training data for testing each bioregion
print("Number of samples taken from training data:", len(df_sample[df_sample.index.isin(df_bioregion_test.index)]))

# Predict all 100k of these bioregion testing datapoints
X = df_bioregion_test.drop(columns=['tree_cover','y', 'x', 'tile_id', 'koppen_class']) 
X_normalized = StandardScaler().fit_transform(X) # Is this scaling consistent across different datasets?

y_test = df_bioregion_test[['tree_cover', 'koppen_class']]
y_pred_percent = model.predict(X_normalized)
y_pred = [percent.argmax() for percent in y_pred_percent]
# print(classification_report(y_test['tree_cover'], y_pred))

# Join predictions with true values and koppen_class
results = df_bioregion_test[['tree_cover', 'koppen_class']].copy()
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
# -




