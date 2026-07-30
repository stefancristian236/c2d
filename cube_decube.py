from astropy.nddata import Cutout2D
import numpy as np

def get_tile_wavelengths(sensor_layout: np.ndarray, 
                         tile_x: int, 
                         tile_y: int):
    
    raw_wl = np.zeros((tile_x, tile_y), dtype=float)
    
    for i in range(tile_x):
        for j in range(tile_y):
            sample = sensor_layout[i::tile_x, j::tile_y]
            raw_wl[i, j] = np.median(sample)

    wl_flatten = raw_wl.flatten()
    
    sorted_wl = np.sort(wl_flatten)
    
    wlidx = np.argsort(np.argsort(wl_flatten)) 
    
    return raw_wl, sorted_wl, wlidx


def cube(image_2c: np.ndarray, 
         sensor_layout: np.ndarray, 
         tile_x: int, 
         tile_y: int):
    
    _, sorted_wl, wlidx = get_tile_wavelengths(sensor_layout, tile_x, tile_y)
    
    cube_size = tile_x * tile_y
    x_side, y_side = image_2c.shape

    x_out = int(np.ceil(x_side / tile_x) * tile_x)
    y_out = int(np.ceil(y_side / tile_y) * tile_y)

    padded_data = np.pad(
        image_2c, 
        ((0, x_out - x_side), (0, y_out - y_side)), 
        mode='constant', 
        constant_values=0
    )
    cube_data = np.zeros((x_side, y_side, cube_size), dtype=float)

    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            flat_idx = idx0 * tile_y + idx1
            z_slice = wlidx[flat_idx]
            
            subgrid = padded_data[idx0::tile_x, idx1::tile_y]
            upsampled = np.repeat(np.repeat(subgrid, tile_x, axis=0), tile_y, axis=1)
            cube_data[:, :, z_slice] = upsampled[:x_side, :y_side]

    return cube_data, sorted_wl


def decube(cube_data: np.ndarray, 
           cube_wl: np.ndarray, 
           sensor_layout: np.ndarray, 
           tile_x: int, 
           tile_y: int):
    
    size_x, size_y, _ = cube_data.shape

    raw_wl, _, _ = get_tile_wavelengths(sensor_layout, tile_x, tile_y)

    band_pos = np.zeros((tile_x, tile_y), dtype=int)
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            band_pos[idx0, idx1] = np.argmin(np.abs(cube_wl - raw_wl[idx0, idx1]))

    image_recon = np.zeros((size_x, size_y), dtype=float)
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            band_idx = band_pos[idx0, idx1]
            image_recon[idx0::tile_x, idx1::tile_y] = cube_data[idx0::tile_x, idx1::tile_y, band_idx]

    return image_recon


def cut_data(data: np.ndarray, 
             center: tuple, 
             size: tuple):
    
    return Cutout2D(data, center, size).data

def run_pipeline(
    mode: str, 
    data: np.ndarray, 
    sensor_layout: np.ndarray, 
    tile_x: int, 
    tile_y: int, 
    cube_wl: np.ndarray | None = None, 
    crop_center: tuple  | None= None, 
    crop_size: tuple | None = None
):
    mode = mode.strip().lower()

    if crop_center is not None and crop_size is not None:
        data = cut_data(data, crop_center, crop_size)
    elif crop_center is not None or crop_size is not None:
        raise ValueError("Both 'crop_center' and 'crop_size' must be provided to crop the data.")
    if mode == 'cube':
        return cube(data, sensor_layout, tile_x, tile_y)

    elif mode == 'decube':
        if cube_wl is None:
            raise ValueError("You must provide 'cube_wl' (wavelengths list) to decube.")

        return decube(data, cube_wl, sensor_layout, tile_x, tile_y)

    else:
        raise ValueError(f"Invalid mode: '{mode}'. Allowed modes are 'cube' or 'decube'.")