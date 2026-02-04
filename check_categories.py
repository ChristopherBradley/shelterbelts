from shelterbelts.utils import create_test_woody_veg_dataset
from shelterbelts.indices import tree_categories
import numpy as np

ds = create_test_woody_veg_dataset()

# Run with different edge_size values
ds1 = tree_categories(ds, stub='e1', outdir='/tmp', plot=False, save_tif=False, edge_size=1, min_patch_size=5)
ds10 = tree_categories(ds, stub='e10', outdir='/tmp', plot=False, save_tif=False, edge_size=10, min_patch_size=5)

print("edge_size=1 categories:", np.unique(ds1['tree_categories'].values))
print("edge_size=10 categories:", np.unique(ds10['tree_categories'].values))

# Check if any patches exist
print("\nWith edge_size=1:")
print(f"  Scattered (11): {(ds1['tree_categories'] == 11).sum().item()}")
print(f"  Core (12): {(ds1['tree_categories'] == 12).sum().item()}")
print(f"  Edge (13): {(ds1['tree_categories'] == 13).sum().item()}")
print(f"  Corridor (14): {(ds1['tree_categories'] == 14).sum().item()}")

print("\nWith edge_size=10:")
print(f"  Scattered (11): {(ds10['tree_categories'] == 11).sum().item()}")
print(f"  Core (12): {(ds10['tree_categories'] == 12).sum().item()}")
print(f"  Edge (13): {(ds10['tree_categories'] == 13).sum().item()}")
print(f"  Corridor (14): {(ds10['tree_categories'] == 14).sum().item()}")
