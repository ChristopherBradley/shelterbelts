# +
# %%time
import os
import glob
import pickle

import numpy as np
import pandas as pd
import rioxarray as rxr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns


def monthly_mosaic(sentinel_file, tree_file=None, clip_percentile=(2, 98)):
    """Create cloud-free monthly sentinel imagery"""

    # Load the sentinel imagery
    print(f"Loading {sentinel_file}")
    with open(sentinel_file, 'rb') as f:
        ds_sentinel = pickle.load(f)  # xarray.Dataset, dims: time, y, x

    # Reproject to match tree mask exactly (should already have roughly the same dimensions)
    if tree_file:
        da_tree = rxr.open_rasterio(tree_file).isel(band=0).drop_vars("band")
        ds_sentinel = ds_sentinel.rio.reproject_match(da_tree)
    else:
        da_tree = None  # Would use this option when applying the model to unseen data
    
    # Create tile_array and timestamps
    bands = list(ds_sentinel.data_vars)
    T = ds_sentinel.dims['time']
    C = len(bands)
    H, W = ds_sentinel.dims['y'], ds_sentinel.dims['x']

    # Stack all bands into shape (H, W, T, C)
    tile_array = np.zeros((H, W, T, C), dtype=np.float32)
    for c, var in enumerate(bands):
        tile_array[:, :, :, c] = ds_sentinel[var].values.transpose(1, 2, 0)  # y,x,time → H,W,T

    # Interpolate and normalise the data
    timestamps = pd.to_datetime(ds_sentinel.time.values)
    monthly_tile = monthly_median_stack(tile_array, timestamps, clip_percentile)

    return monthly_tile, da_tree


def monthly_median_stack(tile_array, timestamps, clip_percentile=(2, 98)):
    """
    Faster monthly median stack with global interpolation.
    Missing months are filled by interpolating the global per-band median time series.
    """
    H, W, T, C = tile_array.shape
    timestamps = pd.to_datetime(timestamps)

    monthly = np.full((H, W, 12, C), np.nan, dtype=np.float32)

    # Step 1: per-month median
    for month in range(1, 13):
        mask = timestamps.month == month
        if not np.any(mask):
            continue
        monthly[:, :, month-1, :] = np.median(tile_array[:, :, mask, :], axis=2)

    # Step 2: interpolate *global* median time series per band
    for c in range(C):
        global_series = np.nanmedian(monthly[:, :, :, c], axis=(0, 1))  # shape (12,)
        idx = np.arange(12)
        valid = ~np.isnan(global_series)
        if np.any(valid):
            filled = np.interp(idx, idx[valid], global_series[valid])
            # fill only NaN months with broadcasted values
            for m in range(12):
                if np.isnan(monthly[0, 0, m, c]):  # month is empty
                    monthly[:, :, m, c] = filled[m]

    # Step 3: clip + normalize
    for c in range(C):
        band = monthly[:, :, :, c]
        vmin, vmax = np.nanpercentile(band, clip_percentile)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        band = np.clip(band, vmin, vmax)
        band = (band - vmin) / (vmax - vmin + 1e-6)
        monthly[:, :, :, c] = band

    monthly_values = np.nan_to_num(monthly, nan=0.0, posinf=1.0, neginf=0.0)
    return monthly_values


def extract_patches(X, y=None, patch_size=64, stride=32):
    """
    Create smaller patches for the convolutional network.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (H, W, C, T) or (H, W, C)
    y : np.ndarray or None
        Shape (H, W). If None, just return X patches.
    patch_size : int
        Size of square patch.
    stride : int
        Step size when sliding the patch window.
        
    Returns
    -------
    patches_X : np.ndarray
        Shape (N, patch_size, patch_size, C, T) or (N, patch_size, patch_size, C)
    patches_y : np.ndarray
        Shape (N, patch_size, patch_size, 1) if y is provided, else None
    """
    patches_X, patches_y = [], []
    H, W = X.shape[:2]

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            Xp = X[i:i+patch_size, j:j+patch_size, ...]
            if y is not None:
                yp = y[i:i+patch_size, j:j+patch_size]
                if np.any(yp):
                    patches_X.append(Xp)
                    patches_y.append(yp[..., None])
            else:
                patches_X.append(Xp)

    patches_X = np.array(patches_X)
    
    if y is not None:
        if len(patches_y) == 0:
            # no patches had trees → return empty array with correct shape
            patches_y = np.zeros((0, patch_size, patch_size, 1), dtype=X.dtype)
        else:
            patches_y = np.array(patches_y)
        return patches_X, patches_y
    else:
        return patches_X, None


def preprocess_tile(sentinel_file, tree_file=None, patch_size=64, stride=32, clip_percentile=(2, 98)):
    """Load, normalise, and reshape the inputs and outputs for a single tile training the CNN
    If tree_file is None, then X_p will be preprocessed as normal, and y_p will also be None
    """
    monthly_tile, ds_tree = monthly_mosaic(sentinel_file, tree_file, clip_percentile)

    # I want to add the option for tree_file to be None, so that I can use the same function when applying the model to unseen data
    tree_values = ds_tree.values if ds_tree is not None else None
    X_p, y_p = extract_patches(monthly_tile, tree_values, patch_size, stride)
    
    # Collapse the time & tile dimensions into a single channel dimension (removes temporal sequence information, but retains temporal variation)
    if len(X_p) > 0:
        X_p = X_p.reshape(
            X_p.shape[0],
            X_p.shape[1], 
            X_p.shape[2],   
            X_p.shape[3] * X_p.shape[4]   # 12 months * 10 bands = 120 channels
        )
    return X_p, y_p

def prep_tiles(sentinel_folder, tree_folder, outdir=".", stub="TEST", patch_size=64, stride=32, clip_percentile=(2, 98), limit=None):
    """Use a bunch of tiles to create the training and testing arrays for the CNN"""

    sentinel_files = sorted(glob.glob(os.path.join(sentinel_folder, "*.pkl")))

    # Initially just trying out 10 tiles as input, around canberra with a good distribution of trees and no trees
    # I should figure out how to balance the larger dataset, and/or adjust the loss function to not overpredict non-trees 
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
    
    sentinel_files = sentinel_files[:limit]
    
    print("Number of tiles to use as input:", len(sentinel_files))

    sentinel_tile_ids = ["_".join(sentinel_tile.split('/')[-1].split('_')[:2]) for sentinel_tile in sentinel_files]
    tree_files = [os.path.join(tree_folder, f'{tile_id}_binary_tree_cover_10m.tiff') for tile_id in sentinel_tile_ids]
    pairs = [(t, s) for t, s in zip(tree_files, sentinel_files)]

    all_X, all_y = [], []
    for tree_file, sentinel_file in pairs:
        X_p, y_p = preprocess_tile(sentinel_file, tree_file, patch_size, stride, clip_percentile)
        all_X.append(X_p)
        all_y.append(y_p)

    X_p = np.concatenate(all_X, axis=0)
    y_p = np.concatenate(all_y, axis=0)

    filename = os.path.join(outdir, f'{stub}_preprocessed.npz')
    np.savez_compressed(filename, X=X_p, y=y_p)
    print(f"Saved: {filename}")
    
    return X_p, y_p

def load_preprocessed_npz(outdir, stub):
    filename = os.path.join(outdir, f'{stub}_preprocessed.npz')
    data = np.load(filename)
    print("Loaded:", filename)
    X, y = data['X'], data['y']
    return X, y
    
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


history_mapping = {'CategoricalAccuracy': 'Training Accuracy', 'val_CategoricalAccuracy': 'Testing Accuracy', 'loss': "Training Loss", 'val_loss': "Testing Loss"}
def loss_plots(history, history_mapping, outdir=".", stub="TEST"):
    
    history_df = pd.DataFrame.from_dict(history.history)
    history_df = history_df.rename(columns = history_mapping)
    
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    
    sns.lineplot(ax=axes[0], data=history_df[['Training Accuracy', 'Testing Accuracy']])
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    
    sns.lineplot(ax=axes[1], data=history_df[['Training Loss', 'Testing Loss']])
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    
    plt.tight_layout()
    filename = os.path.join(outdir, f'{stub}_training_plots.png')
    plt.savefig(filename)

    
def train_model(X, y, outdir=".", stub="TEST", test_size=0.2, random_state=1, batch_size=8, epochs=10):

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = tiny_unet(input_shape=X_train.shape[1:])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    
    # Save the model
    filename = os.path.join(outdir, f'cnn_{stub}.keras')
    model.save(filename)
    print("Saved", filename)

    history_mapping = {'accuracy': 'Training Accuracy', 'val_accuracy': 'Testing Accuracy', 'loss': "Training Loss", 'val_loss': "Testing Loss"}
    loss_plots(history, history_mapping, outdir, stub)
    
    # Create a classification report
    y_pred_prob = model.predict(X_val, batch_size=1)
    y_pred = (y_pred_prob > 0.5).astype(np.uint8)

    y_val_flat = y_val.flatten()
    y_pred_flat = y_pred.flatten()

    print(classification_report(y_val_flat, y_pred_flat, digits=4))
    
    # Should save this classification report to file too

    return model


# -

# %%time
if __name__ == '__main__':

    sentinel_folder = '/scratch/xe2/cb8590/Nick_sentinel'
    tree_folder = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
    limit = 10
    
    outdir = '/scratch/xe2/cb8590/tmp'
    stub = 'CNN_TEST'

    # X, y = prep_tiles(sentinel_folder, tree_folder, outdir, stub, limit=limit)
    X, y = load_preprocessed_npz(outdir, stub)   # Use this if you've already preprocessed the data previously

    model = train_model(X, y, outdir, stub)
    # model = keras.models.load_model(os.path.join(outdir, f'cnn_{stub}.keras')) # Use this if you've already trained the model


def reconstruct_from_patches(patches_y, image_shape, patch_size=64, stride=32):
    """
    Reconstruct full-size image from patches.

    Parameters
    ----------
    patches_y : np.ndarray
        Shape (N, patch_size, patch_size, 1)
    image_shape : tuple
        (H, W) of original image
    patch_size : int
    stride : int

    Returns
    -------
    y_full : np.ndarray
        Shape (H, W)
    """
    H, W = image_shape
    y_full = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)  # for averaging overlapping pixels

    n_patches_per_row = (H - patch_size) // stride + 1
    n_patches_per_col = (W - patch_size) // stride + 1

    idx = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            if idx >= len(patches_y):
                break
            patch = patches_y[idx, ..., 0]  # remove channel dim
            y_full[i:i+patch_size, j:j+patch_size] += patch
            count[i:i+patch_size, j:j+patch_size] += 1
            idx += 1

    # Avoid division by zero
    count[count == 0] = 1
    y_full /= count
    return y_full


# Should probably use this ds as input into monthly_mosaic instead of the filename, so I don't have to load file twice
sentinel_file = '/scratch/xe2/cb8590/Nick_sentinel/g2_017_binary_tree_cover_10m_2020_ds2_2020.pkl'
print(f"Loading {sentinel_file}")
with open(sentinel_file, 'rb') as f:
    ds_sentinel = pickle.load(f)  # xarray.Dataset, dims: time, y, x

# +
# Applying the model to new data
X_p, _, shape = preprocess_tile(s, None)
shape = (ds_sentinel.dims['y'], ds_sentinel.dims['x'])  
y_pred_prob = model.predict(X_val, batch_size=1)
trees_predicted_prob = reconstruct_from_patches(y_pred_prob, shape)
trees_predicted = (trees_predicted_prob > 0.5).astype(np.uint8)
ds_sentinel['trees_predicted'] = ('y', 'x'), trees_predicted_prob
ds_sentinel['trees_predicted'].plot()

# Looks super blocky

# +
# sanity check that the reconstruction is working
tree_file = '/g/data/xe2/cb8590/Nick_Aus_treecover_10m/g2_017_binary_tree_cover_10m.tiff'
da_tree = rxr.open_rasterio(tree_file).isel(band=0).drop_vars("band")
da_tree_reprojected = da_tree.rio.reproject_match(ds_sentinel)
X_p, y_p = preprocess_tile(sentinel_file, tree_file)
reconstructed = reconstruct_from_patches(y_p, da_tree.shape)
plt.imshow(reconstructed)

# Looks fine
