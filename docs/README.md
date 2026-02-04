# Shelterbelts Documentation

This directory contains the Sphinx documentation for the Shelterbelts project.

## Building the Documentation

### Prerequisites

Sphinx and related extensions are typically installed via the project dependencies. If not already installed:

```bash
pip install sphinx sphinx-autodoc-typehints
```

### Building HTML Documentation

From this directory, run:

```bash
make html
```

The built HTML documentation will be in `build/html/`. Open `build/html/index.html` in a web browser to view it.

### Cleaning Build Artifacts

To remove all build artifacts:

```bash
make clean
```

## Documentation Structure

- **source/conf.py**: Sphinx configuration file
- **source/index.rst**: Main documentation entry point
- **source/modules.rst**: API reference documentation
- **Makefile**: Build commands

## Key Features

- **Autodoc**: Automatically extracts and formats docstrings from Python modules
- **NumPy-style Docstrings**: Uses NumPy documentation format with napoleon extension
- **Type Hints**: Includes type hint information in the documentation
- **Autosummary**: Generates summary tables for modules and functions

## Documentation Standards

All code should use NumPy-style docstrings including:

- Brief description
- Parameters with types
- Returns with types
- Notes (optional)
- References (optional)
- Examples (optional)
- See Also (optional)

For more information, see:
- https://www.sphinx-doc.org/
- https://numpydoc.readthedocs.io/
