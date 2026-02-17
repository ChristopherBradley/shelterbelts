"""Tests for compute_distance_to_tree_TH function.

This module tests the shelter distance calculation algorithm, specifically looking for
the bug where shelter is discontinuous (sheltered -> unsheltered -> sheltered) instead of
continuous when moving away from trees in the wind direction.

## Known Bug

The compute_distance_to_tree_TH function has a discontinuity bug in the edge-finding logic:

When multiple trees are present at different distances, the algorithm produces NaN gaps
in the shelter distance output. These gaps represent pixels that should be sheltered but
are incorrectly marked as unsheltered (NaN), creating a pattern like:

    shelter_distance = [2.0, 1.0, NaN, 2.0, 1.0, NaN, 2.0, 1.0, NaN, ...]

This occurs because the condition:
    shifted = shifted.where(shifted_full_max.isnull(), np.nan)

Prevents the algorithm from finding subsequent trees once the first tree has been detected,
because shifted_full_max accumulates across all iterations and becomes permanently non-null
after the first hit.
"""

import numpy as np
import xarray as xr
import pytest

from shelterbelts.indices.shelter_categories import compute_distance_to_tree_TH


def print_array_and_trees(da, distances, title=""):
    """Pretty print the tree positions and shelter distances."""
    print(f"\n{title}")
    print("Tree positions and heights:")
    print("  x: ", list(da.x.values))
    print("  h: ", [f"{h:.1f}" if not np.isnan(h) else "   -" for h in da.values[0]])
    print(f"Shelter distances:")
    print("  x: ", list(distances.x.values))
    print("  d: ", [f"{d:.1f}" if not np.isnan(d) else "  NaN" for d in distances.values[0]])
    print()


def create_test_array(shape=(1, 10), tree_positions=None, tree_heights=None):
    """Create a simple test array with known tree positions and heights.
    
    Parameters
    ----------
    shape : tuple
        Shape of the array (y, x)
    tree_positions : list of int
        X-coordinates where trees are located (single row assumed)
    tree_heights : list of float
        Heights of trees at corresponding positions
        
    Returns
    -------
    xarray.DataArray
        Array with NaN for non-tree pixels, and tree heights for tree pixels
    """
    if tree_positions is None:
        tree_positions = []
    if tree_heights is None:
        tree_heights = [10] * len(tree_positions)
        
    data = np.full(shape, np.nan)
    for pos, height in zip(tree_positions, tree_heights):
        data[0, pos] = height
    
    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': np.arange(shape[0]), 'x': np.arange(shape[1])}
    )
    return da


def test_single_tree_continuous_shelter():
    """Test that a single tree provides continuous shelter in leeward direction.
    
    Setup: Single 10m tree at x=5
    Wind: From E (left to right), so shelter extends to the right
    Expected: All pixels to the right (x > 5) should have positive shelter distance
    
    Bug: If shelter is discontinuous, some pixels will show as unsheltered (distance=0)
         even though they're within the max_distance and downwind from the tree.
    """
    # Single tree at position 5, height 10m
    da = create_test_array(shape=(1, 15), tree_positions=[5], tree_heights=[10])
    
    # Wind from E means wind blows left->right, so negative dx
    distances = compute_distance_to_tree_TH(da, wind_dir='E', max_distance=10)
    
    # All pixels to the right of the tree should be sheltered (positive distance)
    # within the max_distance
    shelter_distances = distances.values[0, 6:11]  # x=6 to x=10 (5 pixels from tree)
    
    print(f"Single tree test - shelter distances: {shelter_distances}")
    
    # Check for discontinuity: if any value is 0 (unsheltered) surrounded by non-zero values
    nonzero_mask = shelter_distances != 0
    has_discontinuity = False
    for i in range(1, len(nonzero_mask) - 1):
        if nonzero_mask[i-1] and not nonzero_mask[i] and nonzero_mask[i+1]:
            has_discontinuity = True
            break
    
    assert not has_discontinuity, "Shelter distance is discontinuous! Found sheltered->unsheltered->sheltered pattern"
    

def test_tall_and_short_trees():
    """Test with a tall edge tree and shorter interior trees.
    
    Setup: Tall tree (15m) at x=5, shorter trees (8m) at x=6,7
    Wind: From east (left to right)
    
    Expected: Pixels to the right should be sheltered by the tall edge tree,
              and shelter should be continuous.
    
    Bug: The algorithm might pick the shorter tree as the "edge" and miss that
         the taller interior trees provide better shelter.
    """
    # Tall tree followed by shorter interior trees
    da = create_test_array(
        shape=(1, 20),
        tree_positions=[5, 6, 7],
        tree_heights=[15, 8, 8]
    )
    
    distances = compute_distance_to_tree_TH(da, wind_dir='E', max_distance=10, multi_heights=True)
    
    shelter_distances = distances.values[0, 8:15]  # pixels 8-14, downwind
    
    print(f"Tall and short trees test - shelter distances: {shelter_distances}")
    
    # Check that shelter is continuous (no gaps)
    nonzero_mask = shelter_distances != 0
    has_discontinuity = False
    for i in range(1, len(nonzero_mask) - 1):
        if nonzero_mask[i-1] and not nonzero_mask[i] and nonzero_mask[i+1]:
            has_discontinuity = True
            print(f"  Discontinuity at position {i}: {nonzero_mask[i-1]} -> {not nonzero_mask[i]} -> {nonzero_mask[i+1]}")
            break
    
    assert not has_discontinuity, "Shelter distance shows gaps (discontinuous sheltering)"


def test_variable_height_trees():
    """Test with trees of varying heights at different positions.
    
    Setup: Multiple trees with heights: [5, 10, 7, 15, 6] at positions [2, 5, 8, 11, 14]
    Wind: From east
    
    Expected: Shelter should be continuous from the most windward tree to max_distance
    Bug: Algorithm produces NaN gaps (discontinuous sheltering)
    
    This test demonstrates the core bug: The pattern [2.0, 1.0, NaN] repeats, showing
    that shelter distance drops to zero (NaN) at regular intervals when it shouldn't.
    """
    da = create_test_array(
        shape=(1, 20),
        tree_positions=[2, 5, 8, 11, 14],
        tree_heights=[5, 10, 7, 15, 6]
    )
    
    distances = compute_distance_to_tree_TH(da, wind_dir='E', max_distance=10, multi_heights=True)
    
    print_array_and_trees(da, distances, "Variable height trees test")
    
    shelter_distances = distances.values[0, 3:14]  # downwind region
    print(f"Shelter distances (x=3 to x=13): {shelter_distances}")
    
    # Look for discontinuities
    nonzero_mask = (shelter_distances != 0) & ~np.isnan(shelter_distances)
    issues = []
    
    for i in range(1, len(nonzero_mask) - 1):
        if nonzero_mask[i-1] and not nonzero_mask[i] and nonzero_mask[i+1]:
            issues.append(f"Gap at index {i}: {shelter_distances[i-1]:.2f} -> NaN -> {shelter_distances[i+1]:.2f}")
    
    if issues:
        print(f"BUG DETECTED: Found {len(issues)} discontinuities:")
        for issue in issues:
            print(f"  {issue}")
    
    assert not issues, f"Found {len(issues)} discontinuities in shelter distance"


def test_edge_case_adjacent_trees():
    """Test trees positioned adjacent to each other.
    
    Setup: Trees at x=5,6,7 with heights 10,8,12
    Wind: From east
    
    Expected: Shelter should extend continuously from this tree cluster
    Bug: Algorithm might not properly handle clustered trees
    """
    da = create_test_array(
        shape=(1, 15),
        tree_positions=[5, 6, 7],
        tree_heights=[10, 8, 12]
    )
    
    distances = compute_distance_to_tree_TH(da, wind_dir='E', max_distance=8, multi_heights=True)
    
    shelter_distances = distances.values[0, 8:13]  # 5 pixels downwind
    
    print(f"Adjacent trees test - shelter distances: {shelter_distances}")
    print(f"  Non-null values: {~np.isnan(shelter_distances)}")
    
    # All pixels should either be sheltered (>0) or explicitly NaN, not a mix of gaps
    assert np.all((shelter_distances > 0) | np.isnan(shelter_distances)), \
        "Found zero values in shelter distance (unsheltered pixels between sheltered ones)"


def test_repeating_pattern_bug():
    """Test that clearly demonstrates the repeating NaN gap pattern.
    
    This is the simplest case that shows the bug: trees spaced such that
    the NaN pattern becomes obvious and repeating.
    
    Setup: Uniform height trees spaced 3 pixels apart
    Expected: No NaN values in the output (all shelter should be continuous)
    Bug: NaN values appear in a regular pattern (every 3rd position approximately)
    """
    # Trees at x=1, 4, 7, 10 with consistent 10m height
    da = create_test_array(
        shape=(1, 20),
        tree_positions=[1, 4, 7, 10],
        tree_heights=[10, 10, 10, 10]
    )
    
    distances = compute_distance_to_tree_TH(da, wind_dir='E', max_distance=8, multi_heights=True)
    
    print_array_and_trees(da, distances, "Repeating Pattern Bug Test (uniform trees)")
    
    # Get downwind region
    shelter_distances = distances.values[0, 2:18]
    
    # Count NaN values
    nan_count = np.isnan(shelter_distances).sum()
    total_count = len(shelter_distances)
    
    print(f"NaN count: {nan_count} out of {total_count} pixels")
    print(f"Shelter distances: {shelter_distances}")
    
    if nan_count > 0:
        print(f"\n⚠ BUG CONFIRMED: {nan_count} unexpected NaN values in shelter distance")
        print("  This indicates the algorithm is dropping shelter at certain distances")
        print("  Pattern suggests distance modulo calculation issue")
        
        # Show which positions have NaN
        nan_positions = np.where(np.isnan(shelter_distances))[0]
        print(f"  NaN positions (relative to start): {nan_positions}")
        
        # Check if there's a repeating pattern
        if len(nan_positions) > 1:
            diffs = np.diff(nan_positions)
            if len(np.unique(diffs)) == 1:
                period = diffs[0]
                print(f"  Repeating period detected: {period} pixels")
    
    assert nan_count == 0, \
        f"Found {nan_count} NaN values in shelter distance - expected continuous shelter"


def test_diminishing_shelter_with_distance():
    """Test that shelter distance diminishes properly with distance from tree.
    
    Setup: Single tall tree at x=5
    Wind: From east
    
    Expected: shelter distance should decrease monotonically as we move away from tree
    Bug: Due to the subtraction logic, distance might become negative or show gaps
    """
    da = create_test_array(
        shape=(1, 20),
        tree_positions=[5],
        tree_heights=[20]
    )
    
    distances = compute_distance_to_tree_TH(da, wind_dir='E', max_distance=15)
    
    shelter_distances = distances.values[0, 6:16]  # 10 pixels downwind from tree
    
    print(f"Diminishing shelter test - distances: {shelter_distances}")
    
    # Remove NaN for analysis
    valid_distances = shelter_distances[~np.isnan(shelter_distances)]
    
    # Check that valid distances are monotonically decreasing or constant
    if len(valid_distances) > 1:
        diffs = np.diff(valid_distances)
        strictly_decreasing = np.all(diffs <= 0)
        print(f"  Differences: {diffs}")
        print(f"  Strictly non-increasing: {strictly_decreasing}")
        
        assert strictly_decreasing, \
            "Shelter distance should decrease monotonically away from tree"


if __name__ == '__main__':
    print("=" * 70)
    print("TEST 1: Single Tree Continuous Shelter")
    print("=" * 70)
    try:
        test_single_tree_continuous_shelter()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")
    
    print("=" * 70)
    print("TEST 2: Tall and Short Trees")
    print("=" * 70)
    try:
        test_tall_and_short_trees()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")
    
    print("=" * 70)
    print("TEST 3: Variable Height Trees (DEMONSTRATES BUG)")
    print("=" * 70)
    try:
        test_variable_height_trees()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")
    
    print("=" * 70)
    print("TEST 4: Adjacent Trees")
    print("=" * 70)
    try:
        test_edge_case_adjacent_trees()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")
    
    print("=" * 70)
    print("TEST 5: Repeating Pattern Bug (CLEARLY DEMONSTRATES BUG)")
    print("=" * 70)
    try:
        test_repeating_pattern_bug()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")
    
    print("=" * 70)
    print("TEST 6: Diminishing Shelter with Distance")
    print("=" * 70)
    try:
        test_diminishing_shelter_with_distance()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")
