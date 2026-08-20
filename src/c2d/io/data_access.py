from pathlib import Path
import numpy as np
from astropy.io import fits
from c2d.analysis import interp as i_c


def extract_parameters(
    asset_dir: str | Path | None = None,
    interp_path: str | Path | None = None,
    not_interp_path: str | Path | None = None,
    sensor_path: str | Path | None = None,
    tile_x: int = 5,
    tile_y: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    extracts 3D spectral data, wavelengths, and sensor layout from FITS files.

    parameters:
        asset_dir: Base directory containing default FITS assets. Defaults to current working directory.
        interp_path: Optional explicit path to the interpolated cube.
        not_interp_path: Optional explicit path to the non-interpolated reference cube.
        sensor_path: Optional explicit path to the sensor wavelength map.
        tile_x, tile_y: Dimensions for block reduction.

    returns:
        inter_cube_data: 3D numpy array of interpolated cube
        reduced_interp_cube: Block-reduced interpolated cube
        not_interp_cube_data: 3D numpy array of original cube
        sensor_data: 2D numpy array of detector layout
        wl: 1D numpy array of wavelengths per spectral layer
    """
    base_dir = Path(asset_dir) if asset_dir is not None else Path.cwd()

    # resolve paths relative to base_dir if explicit paths are not provided
    interp_path = Path(interp_path) if interp_path else base_dir / "HS-L0-FE-025D-05-01_1C_de2.fits"
    not_interp_path = Path(not_interp_path) if not_interp_path else base_dir / "HS-L0-FE-025D-05-01_1C_cube.fits"
    sensor_path = Path(sensor_path) if sensor_path else base_dir / "CRS-HERA-SW101-HyperScout-H-detector-per-pixel-central-wavelength-map.fits"

    # open and process the interpolated cube
    with fits.open(interp_path) as hdul:
        inter_cube_data = np.stack([hdu.data for hdu in hdul[1:]], axis=-1)
        wl = np.array([hdu.header["WAVELEN"] for hdu in hdul[1:]])

    # open and process the non-interpolated reference cube
    with fits.open(not_interp_path) as hdul:
        not_interp_cube_data = np.stack([hdu.data for hdu in hdul[1:]], axis=-1)

    # open sensor layout
    with fits.open(sensor_path) as hdul:
        sensor_data = hdul[0].data

    # block reduction
    reduced_interp_cube = i_c.reduce_block_median(
        inter_cube_data, not_interp_cube_data, tile_x, tile_y
    )

    return inter_cube_data, reduced_interp_cube, not_interp_cube_data, sensor_data, wl