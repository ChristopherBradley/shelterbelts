# +
# Train a random forest classifier to predict the woody veg classes, like in Stewart et al. 2025
# -

# Import necessary libraries
import os
import pandas as pd
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
# Start out by training on just 60k samples like in Stewart et al. 2025
sample_size = 60000
random_state = 0
df_sample = df.sample(n=sample_size, random_state=random_state)

X = df_sample.drop(columns=['tree_cover']) # input variables. Could try normalising the y and x and including this as an input variable

X_normalized = (X - X.min()) / (X.max() - X.min()) # Min max normalisation

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
# -

# Determine the most important bands
feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(60) 
top_features
