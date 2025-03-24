# +
# Train a random forest classifier to predict the woody veg classes, like in Stewart et al. 2025
# -

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# %%time
# From my testing, feather seems like the fastest and smallest filetype for a panda dataframe (better than parquet or csv, although csv more readable)
# filename = "/g/data/xe2/cb8590/shelterbelts/woody_veg_preprocessed.feather"
filename = "../data/woody_veg_preprocessed.feather"
df = pd.read_feather(filename) 
df.shape

# +
# Somehow I ended up with 3219 labels = 0.00393701. Need to look into if these should be 0 or NaN. 
df = df[(df['woody_veg'] == 0) | (df['woody_veg'] == 1)]

# Remove any NaN inputs (there are about 7000 of these)
df = df[df.notna().all(axis=1)]

# +
# %%time
# Start out by training on just 60k samples like in Stewart et al. 2025
sample_size = 60000
random_state = 0
df_sample = df.sample(n=sample_size, random_state=random_state)

X = df_sample.drop(columns=['woody_veg', 'y', 'x', 'Unnamed: 0']) # input variables
y = df_sample['woody_veg']  # target variable

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
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
# Took 3 secs to train 10k samples using 32GB RAM
# Took 36 secs to train 60k samples
# Took 4 mins to train 300k samples 

# Based on tuning min_samples_leaf [1,2,5,10,20,40,60], accuracy goes down slightly as min_samples_leaf is increased
# Based on tuning min_samples_splits [1,2,5,6,7,8,9,10,12,14,16,18,20,40,60], accuracy goes up slightly to 5 and down slightly after 10
# Based on tuning n_estimators [10, 50, 100, 200, 300], accuracy goes up slightly to 100 and down slightly after that. Also takes much longer
# Based on tuning sample_size [10, 50, 100, 200], accuracy goes down as sample size is increased if not using randomly distributed values
# Based on tuning sample_size [10, 50, 100, 200], accuracy goes up as sample size is increased if using randomly distributed values

# I should be expecting a validation accuracy of 0.93 based on Stewart et al. 2025. 
# I'm currently getting 93.35% accuracy at sample_size = 60k, or 93.85% accuracy if I increase the sample size to 300k

# Just the 10m pixels: Accuracy goes down to 91%
# Just the temporal columns: Accuracy stays at 92.4%
# Just the focal columns: Accuracy stays at 92.5%
# No vegetation indices: 92.8%
# No NDVI: 93.17%
# No EVI: 92.99%
# Added GRNDVI: 93.39%
# Added BSI: 93.34%
# Just NDVI & EVI: 88%
# No Std: 92.9%
# Top 10 features only: 92.3%
# Removed NaN inputs: 93.46%
# 14 principle components: 91.3%
# 28 principle components: 92.22%
# -

# Determine the most important bands
feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(60)  # Adjust the number as needed

# +
# Standardize the data (important for PCA)
scaler = StandardScaler()
X = df_sample.drop(columns=['woody_veg', 'y', 'x', 'Unnamed: 0']) # input variables
X_scaled = scaler.fit_transform(X)

# Fit PCA, keeping enough components to explain 95% of variance
pca = PCA(n_components=0.99)  # Adjust threshold as needed
X_pca = pca.fit_transform(X_scaled)

# Convert back to DataFrame
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
df_pca.shape
