from pathlib import Path
import numpy as np
from astropy.io import fits
from c2d.analysis import interp as i_c

DEFAULT_ASSET_DIR = Path("/home/stefan/projects/ISS/Trainship/TASKS/TASK3/assets")


def extract_parameters(
    interp_path: str | Path | None = None,
    not_interp_path: str | Path | None = None,
    sensor_path: str | Path | None = None,
    tile_x: int = 5,
    tile_y: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    extracts 3D spectral data, wavelengths, and sensor layout from FITS files.
    
    returns:
        inter_cube_data: 3D numpy array of interpolated cube
        reduced_interp_cube: Block-reduced interpolated cube
        not_interp_cube_data: 3D numpy array of original cube
        sensor_data: 2D numpy array of detector layout
        wl: 1D numpy array of wavelengths per spectral layer
    """
    # fallback to default asset paths if none are explicitly provided
    if interp_path is None:
        interp_path = DEFAULT_ASSET_DIR / "HS-L0-FE-025D-05-01_1C_de2.fits"
    if not_interp_path is None:
        not_interp_path = DEFAULT_ASSET_DIR / "HS-L0-FE-025D-05-01_1C_cube.fits"
    if sensor_path is None:
        sensor_path = DEFAULT_ASSET_DIR / "CRS-HERA-SW101-HyperScout-H-detector-per-pixel-central-wavelength-map.fits"

    # ppen and process the interpolated cube
    with fits.open(interp_path) as hdul:
        # hdul[1:] skips the PrimaryHDU and iterates over ImageHDUs
        inter_cube_data = np.stack([hdu.data for hdu in hdul[1:]], axis=-1)
        wl = np.array([hdu.header["WAVELEN"] for hdu in hdul[1:]])

    # open and process the non-interpolated reference cube
    with fits.open(not_interp_path) as hdul:
        not_interp_cube_data = np.stack([hdu.data for hdu in hdul[1:]], axis=-1)

    # open sensor layout
    with fits.open(sensor_path) as hdul:
        sensor_data = hdul[0].data

    # use the refactored function name: reduce_block_median
    reduced_interp_cube = i_c.reduce_block_median(
        inter_cube_data, not_interp_cube_data, tile_x, tile_y
    )

    return inter_cube_data, reduced_interp_cube, not_interp_cube_data, sensor_data, wl