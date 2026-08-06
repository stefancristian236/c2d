import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from .. import cutout as cut
from .. import plot as plt2

def reduce_interp(interpolated_cube : np.ndarray, 
                  non_interpolated_cube : np.ndarray, 
                  tile_x : int, 
                  tile_y : int) -> np.ndarray:
    
    h, w, bands = interpolated_cube.shape
    small_h, small_w = non_interpolated_cube.shape[0], non_interpolated_cube.shape[1]
    
    pad_h = (small_h * tile_x) - h
    pad_w = (small_w * tile_y) - w
    
    padded_data = np.pad(
        interpolated_cube,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode='constant',
        constant_values=np.nan
    )
    
    reshaped_data = padded_data.reshape(small_h, tile_x, small_w, tile_y, bands)
    reduced_interp_cube = np.nanmedian(reshaped_data, axis = (1, 3))
    
    return reduced_interp_cube

def extract_wl_median(cube : np.ndarray, 
                      wl : np.ndarray, 
                      output_filepath : str) -> np.array:
    
    arr = np.zeros(cube.shape[2])
    
    output_file = open(output_filepath, 'w') if output_filepath else None
    
    try:
        for band in range(cube.shape[2]):
            median_val = np.median(cube[:, :, band])
            if wl is not None:
                line = f'band = {band} / {wl[band]} has median = {median_val}\n'
            else:
                line = f'band = {band} has median = {median_val}\n'
            print(line, end='')
            if output_file:
                output_file.write(line)
                
            arr[band] = median_val
    finally:
        if output_file:
            output_file.close()
            
    return arr  

def extract_max_min_wl_value(wl : np.array, 
                             cube : np.array) -> tuple:
    
    band_maxs = np.nanmax(cube, axis = (0, 1))
    band_mins = np.nanmin(cube, axis = (0, 1))
    max_idx = np.argmax(band_maxs)
    min_idx = np.argmin(band_mins)
    return wl[max_idx], wl[min_idx], max_idx, min_idx

def layer_analysis_mean (cube : np.array) -> np.array:
    per_layer_mean = np.mean(
        cube, axis = (0, 1)
    )
    return np.sum(cube > per_layer_mean, axis = (0, 1))    



## given a pixel section of 3 x 3
## extract a star pattern around the middle target pixel
## calculate the median of it
##
## 0 1 2        
## 3 4 5  ->  2 4 6 8 
## 6 7 8
def star_pattern(section: np.ndarray) -> float:
    if section.shape != (3, 3):
        raise ValueError('the section needs to be a 3x3 square!')
    star = [
        section[0, 1],
        section[1, 0],
        section[1, 2],
        section[2, 1],
        section[1, 1]
    ]
    return float(np.median(star))
    
##extracts the values around the center and calculates the median

def square_pattern(section: np.ndarray) -> float:
    if section.shape != (3, 3):
        raise ValueError('the section needs to be a 3x3 square!')
    outer_square = np.delete(section, 4)
    return float(np.median(outer_square)) 

#extract the pixel indices for high reflectance pixels
def find_hotPixels(section: np.ndarray) -> tuple[int, int]:
    val = np.max(section)
    indices = np.argwhere(section == val)
    return indices

#extract the pixel indices for low reflectance pixels
def find_coldPixels(section: np.ndarray) -> tuple[int, int]:
    val = np.min(section)
    indices = np.argwhere(section == val)
    return indices

# calculate the median value according to the interpolation algorithm
# median of ((i + 1, j, k[i + 1, j]), (i, j + 1, k[i, j + 1]), (i - 1, j, k[i - 1, j]), (i, j - 1, k[i, j - 1]))
# i++ -> -5, i-- -> +5, j++ -> +5, j-- -> -5

def neighbours_compare(section_cube: np.ndarray, wl_matrix: np.ndarray) -> float:
    h, w, bands = section_cube.shape
    values = np.zeros(4)
    if section_cube.shape != (3, 3):
        raise ValueError('the section needs to be a 3x3 square')
    
    
def plot_med_interVSnotInterp(interp, not_interp, wl_arr):
    interp_median = np.median(
        interp, axis = (0, 1)
    )
    not_interp_median = np.median(
        not_interp, axis = (0, 1)
    )
    
    plt.plot(wl_arr, 
         interp_median,
         'o--',
         color = 'red',
         label = 'interpolated'
         )
    plt.plot(wl_arr, 
         not_interp_median,
         'o--',
         color = 'black',
         label = 'not interpolated'
         )

    plt.grid(alpha = .5)
    plt.legend()
    plt.show()
    
def print_corrleation(arr1, arr2, wl):
    for band in range(25):
        corr = np.corrcoef(arr1[:, :, band], arr2[:, :, band])
        print(f'{wl[band]} -> {corr[0, 1]}')
        
def pip1(non_inter_cube, inter_cube, center, size, wl_arr):
    zone2_interp = cut.cut_cube(inter_cube, center, size)
    zone2 = cut.cut_cube(non_inter_cube, center, size)
    print_corrleation(zone2, zone2_interp, wl_arr)
    plot_med_interVSnotInterp(zone2_interp, zone2, wl_arr)
    
    return zone2_interp, zone2
    
    
def med_per_band(cube):
    medians = np.zeros(cube.shape[2])
    for band in range(cube.shape[2]):
        medians[band] = star_pattern(cube[:,:, band])
    return medians


def aproximate_distribution(interpolated_cube, not_interpoalted_cube, wl):
    inter_median = np.median(
        interpolated_cube, axis = (0, 1)
    )
    median = np.median(
        not_interpoalted_cube, axis = (0, 1)
    )
    diff_med = np.abs(inter_median - median)
    
    plt.hist(diff_med)