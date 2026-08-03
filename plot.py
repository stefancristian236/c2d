import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

########## dict format for input:

#    [{"x": -, "y": -, "label": -, }]


############################################33



#################### apply Savitzky-Golay fitler on data
#################### apply spline interpolation on savgol data 
def smooth_spline(x: np.ndarray, 
                  y: np.ndarray, 
                  window_len=11, 
                  poly_order=2, 
                  smooth_s=0.05, 
                  n_points=1000) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
        
    window_len = min(window_len, len(y) - (1 - len(y) % 2))
    y_smooth = savgol_filter(y, window_len, poly_order)
        
    tck, u = splprep([x, y_smooth], k = 3, s = smooth_s)
    u_fine = np.linspace(0, 1, n_points)
    x_fine, y_fine = splev(u_fine, tck)
        
    return x, y, x_fine, y_fine

##############################################################################################

####### main plotting function
## gets data as input and process with smoothing and ploting the data
def plot_series(ax: plt.Axes, 
                x: np.ndarray, 
                y: np.ndarray, 
                data_points: bool, 
                spline_points: bool, 
                label=None, 
                color=None, 
                scale_x=1.0) -> tuple[np.ndarray, np.ndarray]:
    
    x, y, x_fine, y_fine = smooth_spline(np.asarray(x) * scale_x, y)
    
    if data_points:
        ax.plot(x, y, 'o', color=color, alpha=0.8, label=f'{label} (data)' if label else 'Data') 
    if spline_points:
        ax.plot(x_fine, y_fine, '--', color=color, label=f'{label} (fit)' if label else 'Fit')
    if spline_points == 0 and data_points == 0:
        ax.plot(x_fine, y_fine, '--', color=color)
    return x, y
#################################################################################################

##### main plot generator
##### gets as input a list of dictioanry containg the data going to be plotted
##### list size = 1 -> one plot
##### list size > 1 -> overlapping plots 

#### future note:
    #### add removal of data points -> to only keep the curve
    #### add non smoothing visualisation
    #### implememnt using keywork arguments
def make_plot(datasets: list[dict], 
              title=None, 
              xlabel=None, 
              ylabel=None, 
              figsize=(8, 5), 
              show_data=True,
              show_spline=True) -> tuple[plt.Figure, plt.Axes]:
    
    fig, ax = plt.subplots(figsize=figsize)
    for d in datasets:
        plot_data = d.get('data_points', show_data)
        plot_fit = d.get('spline_points', show_spline)
        
        plot_series(
            ax, 
            d["x"], 
            d["y"], 
            data_points=plot_data, 
            spline_points=plot_fit, 
            label=d.get("label"), 
            color=d.get("color")
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5, color='gray')
    
    if show_data == True or show_spline == True:
        ax.legend()
    return fig, ax

#########################################################################################33

##### if needed plot everything on a different graph instead of overlaying
def make_grid(datasets: list[dict], 
              ncols=2, 
              title=None,
              xlabel=None, 
              ylabel=None, 
              figsize=(12,8), 
              show_data=True, 
              show_spline=True) -> tuple[plt.Figure, np.ndarray]:
    
    nrows = int(len(datasets)) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize = figsize, squeeze= False)
        
    for ax, d in zip(axes.flat, datasets):
        plot_data = d.get('data_points', show_data)
        plot_fit = d.get('spline_points', show_spline)
        
        plot_series(
            ax, 
            d["x"], 
            d["y"], 
            data_points=plot_data, 
            spline_points=plot_fit, 
            label=d.get("label"), 
            color=d.get("color")
        )
        
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