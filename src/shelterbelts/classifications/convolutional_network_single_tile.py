import os
import pickle
import rioxarray as rxr

sample_tree_filename = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/g2_26729_binary_tree_cover_10m.tiff'
sample_sentinel_filename = '/scratch/xe2/cb8590/Nick_sentinel/g2_26729_ds2_2017.pkl'

# %%time
ds_tree = rxr.open_rasterio(sample_tree_filename).isel(band=0).drop_vars('band')

# %%time
with open(sample_sentinel_filename, 'rb') as file:
    ds_sentinel = pickle.load(file)

# %%time
ds_sentinel = ds_sentinel.rio.reproject_match(ds_tree)

ds_sentinel

import os
import pickle
import numpy as np
import rioxarray as rxr
import xarray as xr
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model

# +
# -------------------
# 2. Convert to numpy
# -------------------
# Stack time+band into a "channel" dimension
arr = xr.concat([ds_sentinel[var] for var in ds_sentinel.data_vars], dim="band")
# arr dims: band=10, time=39, y=201, x=201

arr = arr.transpose("y", "x", "time", "band").values  # (201,201,39,10)
H, W, T, B = arr.shape
X_full = arr.reshape(H, W, T * B)  # (201,201,390)

y_full = ds_tree.values.astype("float32")  # (201,201)


# +
# -------------------
# 3. Extract patches
# -------------------
def extract_patches(X, y, patch_size=64, stride=64, max_patches=50):
    patches_X, patches_y = [], []
    h, w, c = X.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            Xp = X[i:i+patch_size, j:j+patch_size, :]
            yp = y[i:i+patch_size, j:j+patch_size]
            if yp.sum() == 0:  # skip all-zero patches
                continue
            patches_X.append(Xp)
            patches_y.append(yp[..., None])  # add channel
            if len(patches_X) >= max_patches:
                return np.array(patches_X), np.array(patches_y)
    return np.array(patches_X), np.array(patches_y)

X_p, y_p = extract_patches(X_full, y_full, patch_size=64, stride=64, max_patches=100)
print("Patch shapes:", X_p.shape, y_p.shape)  # (N,64,64,390), (N,64,64,1)

# -

# -------------------
# 4. Train/val split
# -------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_p, y_p, test_size=0.2, random_state=42
)


# +
# -------------------
# 5. Define tiny U-Net
# -------------------
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return x

def tiny_unet(input_shape):
    inputs = layers.Input(input_shape)
    c1 = conv_block(inputs, 16); p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1, 32);     p2 = layers.MaxPooling2D((2,2))(c2)
    bn = conv_block(p2, 64)
    u2 = layers.UpSampling2D((2,2))(bn); u2 = layers.Concatenate()([u2,c2]); c3 = conv_block(u2,32)
    u1 = layers.UpSampling2D((2,2))(c3); u1 = layers.Concatenate()([u1,c1]); c4 = conv_block(u1,16)
    outputs = layers.Conv2D(1,1,activation="sigmoid")(c4)
    return Model(inputs, outputs)

model = tiny_unet(input_shape=X_train.shape[1:])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
# -

# %%time
# -------------------
# 6. Train
# -------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=2,  # small because CPU
    epochs=3
)


# +


# -------------------
# 7. Evaluate
# -------------------
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation accuracy:", val_acc)

