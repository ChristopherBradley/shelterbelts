# Notebooks

This directory contains Jupyter notebooks demonstrating the shelterbelts package.

## Structure

### Main Pipeline
- **`pipeline_overview.ipynb`**: Complete workflow using default parameters
  - Shows all 5 pipeline stages end-to-end
  - Uses default values for quick demonstration
  - Good starting point for new users

### Detailed Parameter Demos (.py notebooks)
- **`indices/demo_tree_categories.py`**: Explore all tree categorization parameters
- **`indices/demo_shelter_categories.py`**: Explore shelter identification parameters
- **`indices/demo_buffer_categories.py`**: Explore buffer zone parameters
- **`indices/demo_cover_categories.py`**: Explore land cover classification parameters

Each demo notebook:
- Mirrors the structure of corresponding test files
- Shows side-by-side parameter comparisons
- Includes narrative explanations of parameter effects
- Uses test data for reproducibility

## Optional: Convert to .ipynb later

You can convert any `.py` notebook to `.ipynb` later using [Jupytext](https://jupytext.readthedocs.io/):

```bash
# Convert a .py notebook to .ipynb
jupytext --to ipynb indices/demo_tree_categories.py
```

**Benefits of .py notebooks:**
- Git-friendly: `.py` files show clean diffs
- No binary blobs in version control
- Editable in any text editor
- Auto-syncs when opening in Jupyter

## Git Configuration

Add to `.gitignore`:
```
# Keep .py notebooks, ignore generated .ipynb
notebooks/**/demo_*.ipynb
notebooks/.ipynb_checkpoints/
```

The `.py` files are the source of truth for detailed demos.
