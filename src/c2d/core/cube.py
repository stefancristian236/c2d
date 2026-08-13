"""
a set of functions meant to automate the process of cubing and decubing
snapshot-mosaic sensor data.

- get_tile_wavelengths : extract the per-tile-position wavelength ordering
                          from the sensor's mosaic layout
- cube                 : pack a 2D mosaic image into a 3D (h, w, wl) cube,
                          nearest-neighbour upsampling each band to full res
- decube               : inverse of cube — unpack a 3D cube back into the
                          original 2D mosaic image
- round_trip_check     : cube() then decube() an image and report the
                          reconstruction error (debugging / validation aid)
"""

from typing import NamedTuple

import numpy as np


class TileLayout(NamedTuple):
    """
    per-tile-position wavelength info for a mosaic sensor layout.
    """
    raw_wl: np.ndarray     # (tile_x, tile_y) median wavelength at each tile position
    sorted_wl: np.ndarray  # raw_wl flattened and sorted ascending
    wlidx: np.ndarray      # flat tile position -> rank in sorted_wl (i.e. band index)


def get_tile_wavelengths(sensor_layout: np.ndarray, tile_x: int, tile_y: int) -> TileLayout:
    """
    given the sensor's per-pixel wavelength layout (periodic with period
    tile_x rows / tile_y cols), extract the wavelength assigned to each
    position within one tile, plus the band ordering (ascending
    wavelength) used by cube()/decube().
    """
    h, w = sensor_layout.shape
    
    h_trim = h - (h % tile_x)
    w_trim = w - (w % tile_y)
    
    trimmed_layout = sensor_layout[:h_trim, :w_trim]

    reshaped = trimmed_layout.reshape(h_trim // tile_x, tile_x, w_trim // tile_y, tile_y)
    
    raw_wl = np.median(reshaped, axis=(0, 2))

    wl_flatten = raw_wl.flatten()
    sorted_wl = np.sort(wl_flatten)
    wlidx = np.argsort(np.argsort(wl_flatten))

    return TileLayout(raw_wl, sorted_wl, wlidx)


def cube(
    image_2d: np.ndarray,
    sensor_layout: np.ndarray,
    tile_x: int,
    tile_y: int,
    layout: TileLayout | None = None,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    pack a 2D mosaic image into a 3D (h, w, tile_x*tile_y) cube at
    native macropixel resolution. returns (cube, sorted_wl).
    """
    if layout is None:
        layout = get_tile_wavelengths(sensor_layout, tile_x, tile_y)
    _, sorted_wl, wlidx = layout

    x_side, y_side = image_2d.shape
    cube_size = tile_x * tile_y

    x_out = int(np.ceil(x_side / tile_x) * tile_x)
    y_out = int(np.ceil(y_side / tile_y) * tile_y)

    padded_data = np.pad(
        image_2d,
        ((0, x_out - x_side), (0, y_out - y_side)),
        mode="constant",
        constant_values=fill_value,
    )
    #extract the cube sides
    mx, my = x_out // tile_x, y_out // tile_y
    packed_cube = np.zeros((mx, my, cube_size), dtype=float)

    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            flat_idx = idx0 * tile_y + idx1
            z_slice = wlidx[flat_idx]

            subgrid = padded_data[idx0::tile_x, idx1::tile_y]
            packed_cube[:, :, z_slice] = subgrid

    #crop the mactropixel - filter out padding values
    mx_valid = int(np.ceil(x_side / tile_x))
    my_valid = int(np.ceil(y_side / tile_y))

    return packed_cube[:mx_valid, :my_valid, :], sorted_wl


def decube(
    cube_data: np.ndarray,
    cube_wl: np.ndarray,
    sensor_layout: np.ndarray,
    tile_x: int,
    tile_y: int,
    raw_wl: np.ndarray | None = None,
) -> np.ndarray:
    """
    inverse of cube(): reconstruct the original 2D mosaic image by, for
    each tile position, picking the cube band whose wavelength is closest
    to that position's native wavelength and placing that macropixel-
    resolution band at the matching strided pixels

    `cube_data` is expected at native macropixel resolution (mx, my, bands),
    as produced by cube()
    """
    mx, my, _ = cube_data.shape
    if raw_wl is None:
        raw_wl, _, _ = get_tile_wavelengths(sensor_layout, tile_x, tile_y)
    flat_raw = raw_wl.flatten()

    # match wavelengths
    match_dist = np.abs(cube_wl[:, None] - flat_raw[None, :])
    band_pos = np.argmin(match_dist, axis=0)
    band_pos = band_pos.reshape(tile_x, tile_y)

    # calculate the final image size    
    size_x, size_y = mx * tile_x, my * tile_y
    image_recon = np.zeros((size_x, size_y), dtype=float)

    # perform the reconstruction
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            band_idx = band_pos[idx0, idx1]
            image_recon[idx0::tile_x, idx1::tile_y] = cube_data[:, :, band_idx]

    return image_recon


def round_trip_check(
    image_2d: np.ndarray,
    sensor_layout: np.ndarray,
    tile_x: int,
    tile_y: int,
) -> dict:
    """
    cube then decube `image_2d` and report the reconstruction error.
    """
    layout = get_tile_wavelengths(sensor_layout, tile_x, tile_y)
    raw_wl, sorted_wl, wlidx = layout

    cube_data, sorted_wl = cube(image_2d, sensor_layout, tile_x, tile_y, layout=layout)
    recon = decube(cube_data, sorted_wl, sensor_layout, tile_x, tile_y, raw_wl=raw_wl)

    x_side, y_side = image_2d.shape
    x_valid = min(x_side, recon.shape[0])
    y_valid = min(y_side, recon.shape[1])

    diff = np.abs(
        image_2d[:x_valid, :y_valid].astype(float) - recon[:x_valid, :y_valid]
    )
    return {
        "max_error": float(np.max(diff)),
        "mean_error": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "reconstruction": recon,
    }