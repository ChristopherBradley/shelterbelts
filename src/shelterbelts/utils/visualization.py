"""Visualization utilities for shelterbelts analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import rasterio


def _plot_categories_on_axis(ax, da, colormap, labels, title, legend_inside=False):
    """Helper to plot categorical data on a given axis."""
    worldcover_classes = sorted(colormap.keys())
    present_classes = np.unique(da.values[~np.isnan(da.values)]).astype(int)
    worldcover_classes = [cls for cls in worldcover_classes if cls in present_classes]
    
    colors = [np.array(colormap[k]) / 255.0 for k in worldcover_classes]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        boundaries=[v - 0.5 for v in worldcover_classes] + [worldcover_classes[-1] + 0.5],
        ncolors=len(worldcover_classes)
    )
    
    ax.imshow(da.values, cmap=cmap, norm=norm)
    if title:
        ax.set_title(title, fontsize=30, fontweight='bold')
    ax.axis('off')
    
    if labels:
        legend_elements = [
            Patch(facecolor=np.array(color), label=labels[class_id])
            for class_id, color in zip(worldcover_classes, colors)
        ]
        if legend_inside:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
        else:
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)


def visualise_categories(da, filename=None, colormap=None, labels=None, title=None):
    """Pretty visualisation using a categorical colour scheme."""
    if colormap is None:
        raise ValueError("colormap dictionary must be provided")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    _plot_categories_on_axis(ax, da, colormap, labels, title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        plt.show()


def visualise_categories_sidebyside(da1, da2, filename=None, colormap=None, labels=None, title1=None, title2=None):
    """Display two categorical maps side by side with shared colormap and labels."""
    if colormap is None:
        raise ValueError("colormap dictionary must be provided")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
    _plot_categories_on_axis(ax1, da1, colormap, labels, title1, legend_inside=True)
    _plot_categories_on_axis(ax2, da2, colormap, labels, title2, legend_inside=True)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        plt.show()


def visualise_canopy_height(ds, filename=None):
    """Pretty visualisation of the canopy height.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'canopy_height' variable
    filename : str, optional
        If provided, save the figure to this path
    """
    image = ds['canopy_height']

    # Bin the slope into categories
    bin_edges = np.arange(0, 16, 1) 
    categories = np.digitize(image, bin_edges, right=True)
    
    # Define a color for each category
    colours = plt.cm.viridis(np.linspace(0, 1, len(bin_edges) - 2))
    cmap = ListedColormap(['white'] + list(colours))
    
    # Plot the values
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(categories, cmap=cmap)
    
    # Assign the colours
    labels = [f'{bin_edges[i]}' for i in range(len(bin_edges))]
    labels[-1] = '>=15'
    
    # Place the tick label in the middle of each category
    num_categories = len(bin_edges)
    start_position = 0.5
    end_position = num_categories + 0.5
    step = (end_position - start_position)/(num_categories)
    tick_positions = np.arange(start_position, end_position, step)
    
    cbar = plt.colorbar(im, ticks=tick_positions)
    cbar.ax.set_yticklabels(labels)
    
    plt.title('Canopy Height (m)', size=14)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
        print("Saved:", filename)
    else:
        plt.show()


def tif_categorical(da, filename="output.tif", colormap=None, tiled=False):
    """Save a GeoTIFF with categorical color scheme.
    
    Parameters
    ----------
    da : xarray.DataArray
        The categorical data to save
    filename : str, optional
        Output filename (default: output.tif)
    colormap : dict, optional
        Color map dictionary mapping values to RGB tuples (0-255)
    tiled : bool, optional
        Whether to use tiled compression (default: False)
    """
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=da.shape[0],
        width=da.shape[1],
        count=1,
        dtype="uint8",
        crs=da.rio.crs,
        transform=da.rio.transform(),
        compress="LZW",
        photometric="palette",
        tiled=tiled,
        nodata=da.rio.nodata
    ) as dst:
        dst.write(da.values, 1)
        if colormap:
            dst.write_colormap(1, colormap)
    
    print(f"Saved: {filename}")
