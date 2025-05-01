# +
# Train a random forest classifier to predict the woody veg classes, like in Stewart et al. 2025
# -

# Import necessary libraries
import os
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

outlines_dir = "/g/data/xe2/cb8590/Nick_outlines"
filename = os.path.join(outlines_dir, f"tree_cover_preprocessed.csv")
df = pd.read_csv(filename, index_col=False)
df.shape

# Drop the 174 rows where tree cover values = 2
df = df[(df['tree_cover'] == 0) | (df['tree_cover'] == 1)]
df.shape

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

# Start out by training on just 60k samples like in Stewart et al. 2025
sample_size = 60000
random_state = 1
# df_sample = df.sample(n=sample_size, random_state=random_state)
df_sample_full = df.sample(n=len(df), random_state=random_state)  # randomising everything so I can later use a larger random testing dataset while being sure I don't reuse training data
df_sample = df_sample_full[:sample_size]

df_sample['koppen_class'].value_counts()

# +
# %%time
X = df_sample.drop(columns=['tree_cover','y', 'x', 'tile_id', 'koppen_class']) # input variables. Could try reprojecting and normalising the y and x and including this as an input variable

X_normalized = (X - X.min()) / (X.max() - X.min()) # Min max normalisation. Should probs do this normalisation earlier in the process so I don't need to keep renormalising

y = df_sample['tree_cover']  # target variable

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=random_state)
    
# Parameters same as Stewart et al. 2025
rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=random_state)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%' )
print(classification_report(y_test, y_pred))

# +
# 86% accuracy with 1000 pixels (4 secs)
# 88% accuracy, 80% 1 recall with 60000 pixels (40 secs).
# 88% accuracy, 80% 1 recall with 60000 pixels (40 secs) and normalised.
# 89% accuracy, 81% 1 recall with 60000 pixels (40 secs) and normalised and x and y.

# +
# Evaluate the accuracy for each bioregion using unused data (a larger sample than the designated test data, but also not used in the training data)
df_bioregion_test = df_sample_full[600000:700000]

# Double check we aren't reusing any of the training data for testing each bioregion
assert len(df_sample[df_sample.index.isin(df_bioregion_test.index)]) == 0

# Predict all 100k of these bioregion testing datapoints
X = df_bioregion_test.drop(columns=['tree_cover','y', 'x', 'tile_id', 'koppen_class']) 
X_normalized = (X - X.min()) / (X.max() - X.min()) 
y_test = df_bioregion_test[['tree_cover', 'koppen_class']]
y_pred = rf_classifier.predict(X_normalized)
print(classification_report(y_test['tree_cover'], y_pred))

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
