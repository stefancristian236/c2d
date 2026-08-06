#### a set of functions meant to automate the process of cubbing and decubbing data
#### get_tile_wavelengths - extract all the n numbers if wavelength using the sensor layout of the camera
#### cub - perform the cubbing -> packs all data from a 2d arry into a 3d array h x w x wl
#### decube - perform the inverse of cubing by uncpacking the 3d array into a 2d array
import numpy as np

def get_tile_wavelengths(sensor_layout: np.ndarray, tile_x: int, tile_y: int):
    h, w = sensor_layout.shape
    reshaped = sensor_layout.reshape(h // tile_x, tile_x, w // tile_y, tile_y)
    raw_wl = np.median(reshaped, axis=(0, 2))

    wl_flatten = raw_wl.flatten()
    sorted_wl = np.sort(wl_flatten)
    wlidx = np.argsort(np.argsort(wl_flatten))

    return raw_wl, sorted_wl, wlidx


def cube(image_2c: np.ndarray, sensor_layout: np.ndarray, tile_x: int, tile_y: int):
    _, sorted_wl, wlidx = get_tile_wavelengths(sensor_layout, tile_x, tile_y)
    
    x_side, y_side = image_2c.shape
    cube_size = tile_x * tile_y

    x_out = int(np.ceil(x_side / tile_x) * tile_x)
    y_out = int(np.ceil(y_side / tile_y) * tile_y)

    padded_data = np.pad(
        image_2c, 
        ((0, x_out - x_side), (0, y_out - y_side)), 
        mode='constant', 
        constant_values=0
    )
    
    padded_cube = np.zeros((x_out, y_out, cube_size), dtype=float)

    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            flat_idx = idx0 * tile_y + idx1
            z_slice = wlidx[flat_idx]
            
            subgrid = padded_data[idx0::tile_x, idx1::tile_y]
            upsampled = np.repeat(np.repeat(subgrid, tile_x, axis=0), tile_y, axis=1)
            padded_cube[:, :, z_slice] = upsampled
    return padded_cube[:x_side, :y_side, :], sorted_wl


def decube(cube_data: np.ndarray, cube_wl: np.ndarray, sensor_layout: np.ndarray, tile_x: int, tile_y: int):
    size_x, size_y, _ = cube_data.shape
    raw_wl, _, _ = get_tile_wavelengths(sensor_layout, tile_x, tile_y)
    flat_raw = raw_wl.flatten()
    band_pos = np.argmin(np.abs(cube_wl[:, None] - flat_raw[None, :]), axis=0).reshape(tile_x, tile_y)

    image_recon = np.zeros((size_x, size_y), dtype=float)
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            band_idx = band_pos[idx0, idx1]
            image_recon[idx0::tile_x, idx1::tile_y] = cube_data[idx0::tile_x, idx1::tile_y, band_idx]

    return image_recon