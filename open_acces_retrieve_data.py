import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from . import interp_compare as i_c


def extract_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #import .fits images
    mars_interp_cube_path = '/home/stefan/Projects/ISS/Trainship/TASKS/TASK2/entry3/assets/HS-L0-FE-025D-05-01_1C_de2.fits'
    sensor_layout = '/home/stefan/Projects/ISS/Trainship/TASKS/TASK2/entry3/assets/CRS-HERA-SW101-HyperScout-H-detector-per-pixel-central-wavelength-map.fits'
    mars_not_interp_cube_path = '/home/stefan/Projects/ISS/Trainship/TASKS/TASK2/entry3/assets/HS-L0-FE-025D-05-01_1C_cube.fits'
    #open fits files
    with fits.open(mars_interp_cube_path) as hdul:
        inter_cube_data = np.stack(
            [
                hdul[idx].data for idx in range(1, np.size(hdul[1:]) + 1)
            ], axis = -1
        )
    
        wl = np.stack(
            [
                hdul[idx].header['WAVELEN'] for idx in range(1, np.size(hdul[1:]) + 1)
            ], axis = -1
        )
    
    with fits.open(mars_not_interp_cube_path) as hdul:
        not_interp_cube_data = np.stack(
            [
                hdul[idx].data for idx in range(1, np.size(hdul[1:]) + 1)
            ], axis = -1
        )

    with fits.open(sensor_layout) as hdul:
        sensor_data = hdul[0].data
        
    reduced_interp_cube = i_c.reduce_interp(inter_cube_data, not_interp_cube_data, 5, 5)
    
    return inter_cube_data, reduced_interp_cube, not_interp_cube_data, sensor_data, wl