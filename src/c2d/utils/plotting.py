import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

def smooth_spline(
    x: np.ndarray,
    y: np.ndarray,
    window_len: int = 11,
    poly_order: int = 2,
    smooth_s: float = 0.05,
    n_points: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    applies a Savitzky-Golay filter on data and fits a spline interpolation.
    ansures data is sorted and window lengths meet SciPy's constraints.
    """
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    
    # sort by x to prevent spline overlap artifacts
    order = np.argsort(x_arr)
    x_arr, y_arr = x_arr[order], y_arr[order]
        
    # ensure window_len is odd and <= len(y)
    max_len = len(y_arr) - (1 if len(y_arr) % 2 == 0 else 0)
    window_len = min(window_len, max_len)
    
    # Savitzky-Golay requires window_length > poly_order
    if window_len <= poly_order:
        window_len = poly_order + 1 + (poly_order % 2)

    y_smooth = savgol_filter(y_arr, window_len, poly_order)
        
    tck, u = splprep([x_arr, y_smooth], k=3, s=smooth_s)
    u_fine = np.linspace(0, 1, n_points)
    x_fine, y_fine = splev(u_fine, tck)
        
    return x_arr, y_arr, x_fine, y_fine

def plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    data_points: bool,
    spline_points: bool,
    label: str = None,
    color: str = None,
    scale_x: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    processes and plots a single series of data, optionally showing raw points and/or fits.
    """
    x_arr, y_arr, x_fine, y_fine = smooth_spline(np.asarray(x) * scale_x, y)
    
    if data_points:
        point_label = f"{label} (data)" if label else "Data"
        ax.plot(x_arr, y_arr, 'o', color=color, alpha=0.8, label=point_label) 
        
    if spline_points:
        fit_label = f"{label} (fit)" if label else "Fit"
        ax.plot(x_fine, y_fine, '--', color=color, label=fit_label)
        
    # fallback if both are disabled but function is called
    if not spline_points and not data_points:
        ax.plot(x_fine, y_fine, '--', color=color)
        
    return x_arr, y_arr

def make_plot(
    datasets: list[dict],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple = (8, 5),
    show_data: bool = True,
    show_spline: bool = True,
    **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """
    generates a single plot overlaying multiple datasets.
    
    expected dict format in datasets list:
    {"x": array, "y": array, "label": str, "color": str (optional), "scale_x": float (optional)}
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for d in datasets:
        plot_data = d.get('data_points', show_data)
        plot_fit = d.get('spline_points', show_spline)
        
        plot_series(
            ax=ax,
            x=d["x"],
            y=d["y"],
            data_points=plot_data,
            spline_points=plot_fit,
            label=d.get("label"),
            color=d.get("color"),
            scale_x=d.get("scale_x", 1.0)
        )

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    
    ax.grid(True, alpha=0.5, color='gray')
    
    if show_data or show_spline:
        ax.legend()
        
    return fig, ax

def make_grid(
    datasets: list[dict],
    ncols: int = 2,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple = (12, 8),
    show_data: bool = True,
    show_spline: bool = True
) -> tuple[plt.Figure, np.ndarray]:
    """
    generates a grid of independent subplots, one for each dataset provided.
    """
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
    for ax, d in zip(axes.flat, datasets):
        plot_data = d.get('data_points', show_data)
        plot_fit = d.get('spline_points', show_spline)
        
        plot_series(
            ax=ax,
            x=d["x"],
            y=d["y"],
            data_points=plot_data,
            spline_points=plot_fit,
            label=d.get("label"),
            color=d.get("color"),
            scale_x=d.get("scale_x", 1.0)
        )
        
        if d.get("label"):
            ax.set_title(d.get("label"))
        
        ax.grid(True, alpha=0.5)
        ax.legend()
    
    # hide any leftover empty subplots from the grid math
    for ax in axes.flat[len(datasets):]:
        ax.axis('off') 
        
    if title: fig.suptitle(title)
    if xlabel: fig.supxlabel(xlabel)
    if ylabel: fig.supylabel(ylabel)
    
    fig.tight_layout()
    
    return fig, axes