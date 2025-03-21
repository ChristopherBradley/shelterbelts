# +
# Train a random forest classifier to predict the woody veg classes, like in Stewart et al. 2025
# -

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

random_state = 0

# From my testing, feather seems like the fastest and smallest filetype for a panda dataframe (better than parquet or csv, although csv more readable)
# %%time
filename = "/g/data/xe2/cb8590/shelterbelts/woody_veg_preprocessed.feather"
df = pd.read_feather(filename) 
df.shape

# Somehow I ended up with 3219 labels = 0.00393701. Need to look into if these should be 0 or NaN. 
df['woody_veg'] = round(df['woody_veg'])

# Start out by training on just 60k samples like in Stewart et al. 2025
df_60k = df.iloc[:1000]

X = df_60k.drop(columns=['woody_veg', 'y', 'x', 'Unnamed: 0']) # input variables
y = df_60k['woody_veg']  # target variable

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# +
# %%time
# Parameters same as Stewart et al. 2025
rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=20, min_samples_leaf=10, random_state=random_state)

# Train the model
rf_classifier.fit(X_train, y_train)

# Took 3 secs to train 10k samples using 32GB RAM
# Took 36 secs to train 60k samples using 32GB RAM

# +
# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report for detailed evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

# 94% accuracy with 60k total samples (42k training, 18k testing)
# 94% accuracy with 10k samples, but slightly lower precision and recall for the 0's compared to 1's
# 94% accuracy with 10k samples, but much lower precision and recall for the 0's compared to 1's
# -


