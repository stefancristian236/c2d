import numpy as np
from astropy.nddata import Cutout2D

## performs a cut on every cube layer
## takes as input a cube h x w x l
def cut_cube(data: np.ndarray, 
             center: tuple, 
             size: tuple) -> np.ndarray:
    ## create a blank cube with the same size as size of the region going to be cut x number of wavelengths
    output_cube = np.zeros((size[0], size[1], data.shape[2]))
    
    ## iterate trough the cube 
    ## on every layer cut the same location to extract the coresponding zone on every layer -> extract the macropixels
    for band in range(data.shape[2]):
        ## fill the blank cube with the Cutout data
        output_cube[:, :, band] = Cutout2D(data[:, :, band], center, size).data
        
    ## return in 3d np.array consisting ot the extracted zone     
    return output_cube

###############################################################################################################################################


## performs the cutting procedure on a 2d image

def cut_image(data: np.ndarray,
              center: tuple,
              size: tuple) -> np.ndarray:
    
    return (Cutout2D(data, center, size)).data