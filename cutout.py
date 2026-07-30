import numpy as np
from astropy.nddata import Cutout2D

def cut_cube(data: np.ndarray, 
             center: tuple, 
             size: tuple) -> np.ndarray:
    
    output_cube = np.zeros((size[0], size[1], data.shape[2]))
    
    for band in range(data.shape[2]):
        output_cube[:, :, band] = Cutout2D(data[:, :, band], center, size).data
        
    return output_cube