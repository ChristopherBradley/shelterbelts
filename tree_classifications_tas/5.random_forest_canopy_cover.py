# +
# Train a random forest classifier to predict the woody veg classes, like in Stewart et al. 2025
# -

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, mean_absolute_error


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

random_state = 0

filename = "/g/data/xe2/cb8590/shelterbelts/canopycover_preprocessed.csv"
df = pd.read_csv(filename) 
df = df[df.notna().all(axis=1)]

# +
# %%time
# Start out by training on just 60k samples like in Stewart et al. 2025
sample_size = 60000
df_sample = df.sample(n=sample_size, random_state=random_state)
# df_sample = df

X = df_sample.drop(columns=['canopycover', 'y', 'x', 'Unnamed: 0']) # input variables
y = df_sample['canopycover']  # target variable

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
# Parameters same as Stewart et al. 2025
rf_classifier = RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=random_state)
# -

# %%time
# Train the model
rf_classifier.fit(X_train, y_train)

# +
# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Continuous metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
bias = np.mean(y_pred - y_test)
overall_mean = np.mean(y_test)


# +
# Absolute metrics
threshold = 0.1
y_test_bool = (y_test >= threshold).astype(int)
y_pred_bool = (y_pred >= threshold).astype(int)
accuracy = accuracy_score(y_test_bool, y_pred_bool)

# Print results
# print(f"R²: {r2:.4f}")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"Bias: {bias:.4f}")
# print(f"Overall Mean: {overall_mean:.4f}")

# Using 10% tree cover over predicts trees, but 30% has roughly equal precision and recall (93%)
print(classification_report(y_test_bool, y_pred_bool))

# +
# 60k samples, 100 estimators, 10 min_samples_split: r2 = 0.885, accuracy = 0.894, MAE=0.07, training time: 4 mins 43 secs
