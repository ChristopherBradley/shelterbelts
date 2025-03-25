# +
# Train a neural network to compare with random forest predictions
# -

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# +
# Can't import pytorch in John's geospatenv. Need to use my own env on NCI run locally to use pytorch. 
# import torch
# -

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# %%time
# From my testing, feather seems like the fastest and smallest filetype for a panda dataframe (better than parquet or csv, although csv is more readable)
filename = "/g/data/xe2/cb8590/shelterbelts/woody_veg_preprocessed.feather"
df = pd.read_feather(filename) 
df.shape

# Somehow I ended up with 3219 labels = 0.00393701. Need to look into if these should be 0 or NaN. 
df = df[(df['woody_veg'] == 0) | (df['woody_veg'] == 1)]
df = df[df.notna().all(axis=1)]
# df['woody_veg'] = df['woody_veg'].round()

y_categorical = keras.utils.to_categorical(y, 2)


# +
random_state = 0
sample_size = 10000
df_sample = df.sample(n=sample_size, random_state=random_state)

X = df_sample.drop(columns=['woody_veg', 'y', 'x', 'Unnamed: 0']) # input variables
y = df_sample['woody_veg']  # target variable
y_categorical = keras.utils.to_categorical(y, 2)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=random_state)
# -

# Normalize the features (should probs do this before creating the .feather file)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# +
# Define the neural network model. Should probs try fewer layers initially. Have a look at my code from COMP4660.
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # Binary classification (0 or 1)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# +
# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Training accuracies not increasing at all for some reason. Need to debug.
