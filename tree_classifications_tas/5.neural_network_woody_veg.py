# +
# Train a neural network to compare with random forest predictions
# -

import os
import pandas as pd
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
# From my testing, feather seems like the fastest and smallest filetype for a panda dataframe (better than parquet or csv, although csv is more readable)
filename = "/g/data/xe2/cb8590/shelterbelts/woody_veg_preprocessed.feather"
df = pd.read_feather(filename) 
df.shape

# Remove bad samples (about 3000 samples = 0.3, and about 7000 NaN)
df = df[(df['woody_veg'] == 0) | (df['woody_veg'] == 1)]
df = df[df.notna().all(axis=1)]

# +
random_state = 0
# sample_size = 60000
sample_size = 200000
df_sample = df.sample(n=sample_size, random_state=random_state)
# df_sample = df

# Normalise the input features (should probs do this before creating the .feather file)
X = df_sample.drop(columns=['woody_veg', 'y', 'x', 'Unnamed: 0']) # input variables
X = StandardScaler().fit_transform(X)

y = df_sample['woody_veg']  # target variable
y_categorical = keras.utils.to_categorical(y, 2)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=random_state)
# -

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=5,           # Stop training if val_loss doesn't improve for 5 epochs
    restore_best_weights=True  # Restore the best model weights
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
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Specify learning rate here
model.compile(optimizer=optimizer, loss = 'CategoricalCrossentropy', metrics = 'CategoricalAccuracy')        
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    callbacks=[early_stopping], validation_data=(X_test, y_test))

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['categorical_accuracy', 'val_categorical_accuracy']])
plt.show()
# -

y_pred = model.predict(X_test)
y_pred_categorical = [pred.argmax() for pred in y_pred]

y_test_categorical = [categories.argmax() for categories in y_test]

print(classification_report(y_test_categorical, y_pred_categorical))
