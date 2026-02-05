#!/usr/bin/env python
# coding: utf-8

# # Histogram of canopy height model (CHM)
# # Notebook: load data/g2_26729_chm_res10.tif and plot a histogram of heights
# 
# This notebook opens the CHM GeoTIFF, inspects metadata, computes basic statistics,
# and plots a histogram of the non-nodata heights.

# In[9]:


print("Hello world")


# In[10]:


# 1) Import required libraries
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Display matplotlib plots inline when running the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


# 2) Open GeoTIFF with rasterio
# Try both 'data/...' and '../data/...' so the notebook works from different CWDs
fp_candidates = [os.path.join('data','g2_26729_chm_res10.tif'), os.path.join('..','data','g2_26729_chm_res10.tif')]
for fp in fp_candidates:
    if os.path.exists(fp):
        raster_path = fp
        break
else:
    raise FileNotFoundError(f"CHM file not found. Checked: {fp_candidates}")

print('Using file:', raster_path)

src = rasterio.open(raster_path)


# In[12]:


# 3) Inspect raster metadata and CRS
print('bands (count):', src.count)
print('width, height:', src.width, src.height)
print('crs:', src.crs)
print('transform:', src.transform)
print('dtype:', src.dtypes)
print('nodata:', src.nodatavals)


# In[13]:


# 4) Read raster band into NumPy array and mask nodata
band1 = src.read(1)
nodata = src.nodatavals[0] if src.nodatavals is not None else None
arr = band1.astype(float)
if nodata is not None:
    arr[arr == nodata] = np.nan

# Create masked array of valid values
masked = np.ma.masked_invalid(arr)
valid = masked.compressed()  # 1D array of valid (non-nodata) values
print('Valid pixels:', valid.size)


# In[14]:


# 5) Compute basic statistics
import numpy.ma as ma
if valid.size > 0:
    print('min:', valid.min())
    print('median:', np.median(valid))
    print('mean:', valid.mean())
    print('std:', valid.std())
    print('max:', valid.max())
else:
    print('No valid pixels found')


# In[15]:


# 6) Plot histogram of heights and save to file
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.hist(valid, bins=50, color='tab:green', edgecolor='black')
plt.xlabel('Height (m)')
plt.ylabel('Count')
plt.title('CHM Height Distribution')
plt.grid(alpha=0.3)
out_png = 'chm_histogram.png'
plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print('Saved histogram to', out_png)
plt.show()


# # Notes
# # To run: open this notebook in Jupyter or VS Code and run the cells in order.
# # The code tries both 'data/g2_26729_chm_res10.tif' and '../data/g2_26729_chm_res10.tif' locations.
# 

# In[17]:


# Replace values > 40 m or NaN with median of surrounding pixels (3x3 window)
import scipy.ndimage as ndi

# Prepare array for filtering: ensure NaNs where invalid
arr_proc = arr.copy()
invalid_mask = np.isnan(arr_proc) | (arr_proc > 40)
arr_proc[invalid_mask] = np.nan

# Compute neighborhood median (ignores NaNs because we use np.nanmedian)
filtered = ndi.generic_filter(arr_proc, function=lambda vals: np.nanmedian(vals), size=3, mode='nearest')

# Replace invalid pixels with computed median
filled = np.where(invalid_mask, filtered, arr.astype(float))
replaced_count = np.sum(invalid_mask)
print(f"Replaced {replaced_count} pixels (NaN or >40 m) with neighborhood median")

# Save new GeoTIFF
out_tif = os.path.join('g2_26729_chm_res10_filled.tif')
meta = src.meta.copy()
meta.update(dtype='float32', count=1)
with rasterio.open(out_tif, 'w', **meta) as dst:
    dst.write(filled.astype('float32'), 1)

print('Saved filled CHM to', out_tif)


# In[ ]:




