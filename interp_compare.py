import numpy as np

def reduce_interp(interpolated_cube, non_interpolated_cube, tile_x, tile_y):
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

def extract_wl_median(cube):
    for band in range(cube.shape[2]):
        print(f'band = {band} has median = {np.median(cube[:, :, band])}')
        
