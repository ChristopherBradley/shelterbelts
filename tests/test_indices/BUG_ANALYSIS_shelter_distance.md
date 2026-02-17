# Bug Analysis: compute_distance_to_tree_TH Discontinuity

## Bug Description

The `compute_distance_to_tree_TH` function produces NaN gaps in shelter distance calculations, creating discontinuous shelter patterns instead of continuous ones.

### Observed Behavior

With trees at positions [2, 5, 8, 11, 14] with uniform heights, the output shelter distance shows:

```
x:  [0, 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ...]
d:  [-, -, -, 2, 1, NaN, 2, 1, NaN, 2, 1, NaN, 2, 1, NaN, ...]
```

Pattern: **[2.0, 1.0, NaN] repeating** - shelter exists at some distances but drops to NaN at regular intervals.

### Expected Behavior

Shelter distance should be continuous and monotonically decreasing as distance increases from the nearest tree group. The NaN gaps should not exist.

## Root Cause

The bug is in this line (lines 47-50 in shelter_categories.py):

```python
for d in range(1, distance_threshold + 1):
    shifted = shifted.shift(x=dx, y=dy, fill_value=np.nan)
    shifted = shifted.where(shifted_full_max.isnull(), np.nan)  # <-- BUG HERE
    shifted = shifted.where(shifted > 0, np.nan)
    shifted_full_max = shifted_full_max.where(~shifted_full_max.isnull(), shifted)
```

### Problem with `shifted = shifted.where(shifted_full_max.isnull(), np.nan)`

This condition attempts to mask pixels where we've already found a tree:
- **True (keep shifted)**: Where `shifted_full_max` is NaN → haven't found a tree yet
- **False (set shifted to NaN)**: Where `shifted_full_max` is not NaN → already found a tree

**The Issue**: `shifted_full_max` accumulates across all iterations. Once ANY tree is found at ANY distance from a pixel, `shifted_full_max` becomes permanently non-null for that pixel. This prevents the algorithm from:

1. **Seeing interior trees**: After finding the edge tree, interior trees are never detected
2. **Recalculating distances**: Each pixel gets stuck with its first-found tree, unable to update with potentially better shelter

### Why This Matters

When `multi_heights=True`, the algorithm should consider:
- Edge trees (first encountered)
- Interior trees (if they're taller than the edge tree)

But the current logic prevents detection of interior trees because `shifted_full_max` persists once set.

### Periodicity Pattern

The repeating [2, 1, NaN] pattern occurs because:
1. Trees are spaced 3 pixels apart
2. The algorithm processes distances 1, 2, 3, 4, 5, 6, 7, 8...
3. Distance calculation: `distances = shifted_full_max - new_hits`
4. With 3-pixel spacing and 8-pixel max_distance, this creates periodic interference

## Test Results

### Test 3: Variable Height Trees (3 discontinuities found)
```
Shelter distances (x=3 to x=13): [2.0, 1.0, NaN, 2.0, 1.0, NaN, 2.0, 1.0, NaN, 2.0, 1.0]
Discontinuities:
  Gap at index 2: 1.00 -> NaN -> 2.00
  Gap at index 5: 1.00 -> NaN -> 2.00
  Gap at index 8: 1.00 -> NaN -> 2.00
```

### Test 5: Repeating Pattern Bug (10 NaN gaps found)
```
Shelter distances: [2.0, 1.0, NaN, 2.0, 1.0, NaN, 2.0, 1.0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]

Pattern detected: Repeating period of 3 pixels between NaN gaps
NaN positions: [2, 5, 8, 9, 10, 11, 12, 13, 14, 15]
```

## Solution Approach

The algorithm needs to be restructured to:

1. **Reset the edge-finding logic** after each iteration level (distance d)
2. **Only preserve the closest tree** for each pixel, not accumulate all trees
3. **Separate concerns**: 
   - Part 1: Find the edge tree (stop after first hit)
   - Part 2: Optionally find interior trees (only if taller)
   - Part 3: Calculate distances correctly

Alternative: Use `scipy.ndimage.distance_transform_edt` with proper masking to find distances to nearest tree, then calculate shelter based on that.

## Files Affected

- **Function**: `compute_distance_to_tree_TH` in [shelter_categories.py](src/shelterbelts/indices/shelter_categories.py)
- **Lines**: 40-91 (main distance calculation logic)
- **Test**: [test_shelter_distance.py](tests/test_indices/test_shelter_distance.py)
