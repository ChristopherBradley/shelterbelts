# +
# Train a neural network to compare with random forest predictions
# -

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, mean_absolute_error

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# %%time
filename = "/g/data/xe2/cb8590/shelterbelts/canopycover_preprocessed.csv"
df = pd.read_csv(filename) 
df = df[df.notna().all(axis=1)]

random_state = 0
sample_size = 60000
df_sample = df.sample(n=sample_size, random_state=random_state)

X = df_sample.drop(columns=['canopycover', 'y', 'x', 'Unnamed: 0']) # input variables
X = StandardScaler().fit_transform(X)
y = df_sample['canopycover']  # target variable


# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',   
    patience=5,           
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
    keras.layers.Dense(1, activation='linear')  # Continuous classification instead
])

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss = 'mse', metrics = 'mae')        
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    callbacks=[early_stopping], validation_data=(X_test, y_test))

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['mae', 'val_mae']])
plt.show()
# -

# Make predictions on the test set
y_pred = model.predict(X_test)[:,0]
r2 = r2_score(y_test, y_pred)
threshold = 0.1
y_test_bool = (y_test >= threshold).astype(int)
y_pred_bool = (y_pred >= threshold).astype(int)
accuracy = accuracy_score(y_test_bool, y_pred_bool)
print(f"R²: {r2:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# +
# First training w/ 60k samples (2 mins): 17 epochs, 0.08 MAE, 91.9% accuracy
