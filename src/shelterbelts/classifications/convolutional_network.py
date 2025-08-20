# %%time
import os
import glob
import pickle
import numpy as np
import rioxarray as rxr
import xarray as xr
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
from sklearn.metrics import classification_report


# -------------------
# 1. File discovery
# -------------------
sentinel_dir = '/scratch/xe2/cb8590/Nick_sentinel'
sentinel_files = sorted(glob.glob(os.path.join(sentinel_dir, "*.pkl")))

# Initially just trying out 10 tiles as input, around canberra with a good distribution of trees and no trees
interesting_tile_ids = [
"g2_017_",
"g2_019_",
"g2_021_",
"g2_21361_",
"g2_23939_",
"g2_23938_",
"g2_09_",
"g2_2835_",
"g2_25560_",
"g2_24903_"
]
sentinel_files = [filename for filename in sentinel_files if any(tile_id in filename for tile_id in interesting_tile_ids)]

sentinel_tile_ids = ["_".join(sentinel_tile.split('/')[-1].split('_')[:2]) for sentinel_tile in sentinel_files]
tree_files = [f'/g/data/xe2/cb8590/Nick_Aus_treecover_10m/{tile_id}_binary_tree_cover_10m.tiff' for tile_id in sentinel_tile_ids]
pairs = [(t, s) for t, s in zip(tree_files, sentinel_files)]


# +
# pairs = pairs[:10]
# -

def monthly_mosaic(tree_file, sentinel_file):
    # -------------------
    # 1. Load tree mask
    # -------------------
    ds_tree = rxr.open_rasterio(tree_file).isel(band=0).drop_vars("band")

    # -------------------
    # 2. Load Sentinel data
    # -------------------
    print(f"Loading {sentinel_file}")
    with open(sentinel_file, 'rb') as f:
        ds_sentinel = pickle.load(f)  # xarray.Dataset, dims: time, y, x

    # Reproject to match tree mask
    ds_sentinel = ds_sentinel.rio.reproject_match(ds_tree)
    
    # -------------------
    # 3. Create tile_array and timestamps
    # -------------------
    bands = list(ds_sentinel.data_vars)
    T = ds_sentinel.dims['time']
    C = len(bands)
    H, W = ds_sentinel.dims['y'], ds_sentinel.dims['x']

    # Stack all bands into shape (H, W, T, C)
    tile_array = np.zeros((H, W, T, C), dtype=np.float32)
    for c, var in enumerate(bands):
        tile_array[:, :, :, c] = ds_sentinel[var].values.transpose(1, 2, 0)  # y,x,time → H,W,T

    # Create list of acquisition timestamps (assuming ds_sentinel.time contains pd.Timestamp)
    timestamps = pd.to_datetime(ds_sentinel.time.values)

    monthly_tile = monthly_median_stack(tile_array, timestamps)

    return monthly_tile, ds_tree


def monthly_median_stack(tile_array, timestamps, clip_percentile=(2, 98)):
    """
    Convert variable-length Sentinel-2 time series into 12 monthly medians.
    
    Parameters
    ----------
    tile_array : np.ndarray
        Shape (H, W, T, C) where T = number of acquisitions, C = bands.
    timestamps : list-like
        Length T, acquisition datetimes (pd.Timestamp or str).
    clip_percentile : tuple
        Percentiles to use for outlier clipping, e.g. (2, 98).
    
    Returns
    -------
    np.ndarray
        Shape (H, W, 12, C), monthly medians normalized to [0,1].
    """
    H, W, T, C = tile_array.shape
    timestamps = pd.to_datetime(timestamps)
    
    # Prepare monthly result
    monthly = np.full((H, W, 12, C), np.nan, dtype=np.float32)

    for month in range(1, 13):
        mask = timestamps.month == month
        if not np.any(mask):
            continue  # no acquisitions this month
        # Median over all acquisitions in that month
        monthly[:, :, month-1, :] = np.median(tile_array[:, :, mask, :], axis=2)

    # Handle outlier clipping + normalization per band
    for c in range(C):
        band = monthly[:, :, :, c]
        vmin, vmax = np.nanpercentile(band, clip_percentile)
        band = np.clip(band, vmin, vmax)
        # Normalize to [0,1]
        band = (band - vmin) / (vmax - vmin + 1e-6)
        monthly[:, :, :, c] = band

    return monthly


# +
def extract_patches(X, y, patch_size=64, stride=32):
    patches_X, patches_y = [], []
    H, W, _, _ = X.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            Xp = X[i:i+patch_size, j:j+patch_size, :, :]
            yp = y[i:i+patch_size, j:j+patch_size]
            if np.any(yp):
                patches_X.append(Xp)
                patches_y.append(yp[..., None])

    return np.array(patches_X), np.array(patches_y)

# X_p, y_p = extract_patches(monthly_tile, ds_tree.values, patch_size=64, stride=32)
# print(X_p.shape, y_p.shape)  # (#patches, 64, 64, 12*C), (#patches, 64, 64, 1)



# -

# %%time
# -------------------
# 3. Build dataset
# -------------------
all_X, all_y = [], []
for tree_file, sentinel_file in pairs:
    monthly_tile, ds_tree = monthly_mosaic(tree_file, sentinel_file)
    X_p, y_p = extract_patches(monthly_tile, ds_tree.values, patch_size=64, stride=32)
    
    # Collapse the time & tile dimensions into a single channel dimension (removes temporal sequence information, but retains temporal variation)
    X_p_reshaped = X_p.reshape(
        X_p.shape[0],   # number of patches
        X_p.shape[1],   # 64
        X_p.shape[2],   # 64
        X_p.shape[3] * X_p.shape[4]   # 12 months * 10 tiles = 120 channels
    )
    if len(X_p) > 0:
        all_X.append(X_p_reshaped)
        all_y.append(y_p)


X_p = np.concatenate(all_X, axis=0)
y_p = np.concatenate(all_y, axis=0)
print("Final dataset:", X_p.shape, y_p.shape)

# +
# -------------------
# 4. Train/val split
# -------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_p, y_p, test_size=0.2, random_state=42
)

# -------------------
# 5. Tiny U-Net (same as before)
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

# -

# %%time
# -------------------
# 6. Train
# -------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)


# +
# -------------------
# 1. Check distribution in training/validation
# -------------------
def pixel_distribution(y_array):
    # y_array shape: (N, H, W, 1)
    counts = np.bincount(y_array.flatten().astype(int))
    zeros = counts[0] if len(counts) > 0 else 0
    ones  = counts[1] if len(counts) > 1 else 0
    total = zeros + ones
    print(f"Total pixels: {total}, zeros: {zeros} ({zeros/total:.2%}), ones: {ones} ({ones/total:.2%})")

print("Training set distribution:")
pixel_distribution(y_train)

print("Validation set distribution:")
pixel_distribution(y_val)


# +
# -------------------
# 2. Make predictions on validation set
# -------------------
y_pred_prob = model.predict(X_val, batch_size=1)
y_pred = (y_pred_prob > 0.5).astype(np.uint8)

# -------------------
# 3. Flatten for classification report
# -------------------
y_val_flat = y_val.flatten()
y_pred_flat = y_pred.flatten()

print("\nClassification report (pixelwise):")
print(classification_report(y_val_flat, y_pred_flat, digits=4))

# -


