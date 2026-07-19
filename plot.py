import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import BSpline, splprep, splev

#    [{"x": -, "y": -, "label": -, },


def smooth_spline(x, y, window_len=11, poly_order=2, smooth_s=0.05, n_points=500):
        order = np.argsort(x)
        x, y = np.asarray(x)[order], np.asarray(y)[order]
        
        window_len = min(window_len, len(y) - (1 - len(y) % 2))
        y_smooth = savgol_filter(y, window_len, poly_order)
        
        tck, u = splprep([x, y_smooth], k = 3, s = smooth_s)
        u_fine = np.linspace(0, 1, n_points)
        x_fine, y_fine = splev(u_fine, tck)
        
        return x, y, x_fine, y_fine

def plot_series(ax, x, y, label=None, color=None, scale_x=1.0, **smooth_kwargs):
    x, y, x_fine, y_fine = smooth_spline(np.asarray(x) * scale_x, y, **smooth_kwargs)
    ax.plot(x, y, 'o', color=color, alpha=0.8, label=f'{label} (data)' if label else 'Data')
    ax.plot(x_fine, y_fine, '--', color=color, label=f'{label} (fit)' if label else 'Fit')
    return x, y

def make_plot(datasets, title=None, xlabel=None, ylabel=None, figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    for d in datasets:
        plot_series(ax, d["x"], d["y"], label=d.get("label"), color=d.get("color"))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5, color='gray')
    ax.legend()
    return fig, ax

def make_grid(datasets, ncols=2, title=None, xlabel=None, ylabel=None, figsize=(12,8)):
        nrows = -(-len(datasets) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize, squeeze=False)
        
        for ax, d in zip(axes.flat, datasets):
                plot_series(ax, d["x"], d["y"], label=d.get("label"), color=d.get("color"))
                ax.set_title(d.get("label"))
                ax.grid(True, alpha=0.5)
                ax.legend()
    
        for ax in axes.flat[len(datasets):]:
                ax.axis('off') 
        
        fig.suptitle(title)
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.tight_layout()
        
        return fig, axes