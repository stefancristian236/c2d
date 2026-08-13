"""
    a set of functions meant to automate the process of plotting the spectra
    of a given hyperspectral cube
"""

import os
import numpy as np
import c2d.utils.plotting as plt2
import c2d.core.cube as cb
import c2d.core.cutout as ct

def dirs(file_name):
    if os.path.exists(file_name) == False:
        os.mkdir(file_name)


def spectra_pipeline(
    cubed_image : np.ndarray[float, float, float],
    wl : np.ndarray[float],
    cut_ct : tuple[int, int],
    cut_size : tuple[int, int],
    title : str = None,
    tile_x : int = 5,
    tile_y : int = 5
    ):
    
    """
    using the ct and size tuple it slices the targeted area
    performs the median on (0, 1) axis
    cube is allready sorted based on wl
    returns the fig and ax + saves the plot
    defaults to a 5 x 5 grid
    """
    
    # run directory verification
    dirs(f'{title}_spectra')
    
    
    # slice the cube
    cut_cube = ct.cut_cube(cubed_image, (cut_ct[0] // tile_x, cut_ct[1] // tile_y), (cut_size[0] // tile_x, cut_size[1] // tile_y))
    
    # extract the median of each layer
    median_refl = np.median(cut_cube, axis = (0, 1))
    median_refl = median_refl / np.max(median_refl)
    
    # create the cargo for the plotting function
    cargo = {
        'x' : wl,
        'y' : median_refl,
        'label' : title
    }
    
    # call the custom plot function
    fig, ax = plt2.make_plot([cargo],
                             title = title,
                             xlabel = 'Wavelength',
                             ylabel = 'Normalised Reflectance')
    
    fig.savefig(f'{title}_spectra/{title}.png', dpi = 1000)
    
    return fig, ax