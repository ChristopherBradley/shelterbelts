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

# +
# %%time
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
random_state = 0
sample_size = 60000
df_sample = df.sample(n=sample_size, random_state=random_state)

# Normalise the input features (should probs do this before creating the .feather file)
X = df_sample.drop(columns=['tree_cover', 'y', 'x']) # input variables
X = StandardScaler().fit_transform(X)

y = df_sample['tree_cover']  # target variable
y_categorical = keras.utils.to_categorical(y, 2)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=random_state)
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
