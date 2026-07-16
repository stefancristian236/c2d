from astropy.io import fits
from astropy.nddata import Cutout2D
import numpy as np

###############################################################################################################
###############################################################################################################

## extract the needed wavengths and indices for cubing and decubing data
## wl_layout = path to the .fits file
## tile_x and tile_y, size of wavelengths matrix
## return 2 numpy arrays

def extract_wl_cut(wl_layout, tile_x, tile_y, center, size):
    with fits.open(wl_layout) as hdul:
        wl_data = hdul[0].data

    sensor_layout = Cutout2D(wl_data, center, size)
    wl_data = sensor_layout.data
    
    raw_wl = np.zeros((tile_x, tile_y))
    wl_data = np.asarray(wl_data)
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            sample = wl_data[idx0::5, :][:, idx1::5]
            raw_wl[idx0, idx1] = np.median(sample)

    wl_flatten = raw_wl.flatten()
    order = np.argsort(wl_flatten)
    wlidx = np.argsort(order)  
    wl = wl_flatten[wlidx]
    return wl, wlidx, raw_wl


def extract_wl(wl_layout, tile_x, tile_y):

    # extract the wave
    with fits.open(wl_layout) as hdul:
        wl_data = hdul[0].data

    raw_wl = np.zeros((tile_x, tile_y))
    wl_data = np.asarray(wl_data)
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            sample = wl_data[idx0::5, :][:, idx1::5]
            raw_wl[idx0, idx1] = np.median(sample)

    wl_flatten = raw_wl.flatten()
    order = np.argsort(wl_flatten)
    wlidx = np.argsort(order)  
    wl = wl_flatten[wlidx]
    return wl, wlidx, raw_wl


##############################################################################################################
#############################################################################################################

## parse the image on wavelengths
## build a multispectral cube 
## fits_path - path to the .fits image
## wl - an numpy array of wavelengths
## wlidx - the coresponding place
## tile_x, tile_y - side sizes of a wavelengths matrix


def cube(image_2c, wl, wlidx, tile_x : int, tile_y : int, header):
    # total number of != wavelenghts:

    cube_size = tile_x * tile_y


    # acces the data and header information:

    x_side = image_2c.shape[0]
    y_side = image_2c.shape[1]

    x_out = np.ceil(x_side / tile_x) * tile_x
    y_out = np.ceil(y_side / tile_y) * tile_y

    x_pad = int(x_out - x_side)
    y_pad = int(y_out - y_side)

    slice_x = int(x_out // tile_x)
    slice_y = int(y_out // tile_y)

    cube = np.zeros((slice_x, slice_y, cube_size), dtype=float)
    padded_data = np.pad(np.array(image_2c), ((0, x_pad), (0, y_pad)), 'constant', constant_values=0)

    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            sample = wlidx[idx0 * tile_y + idx1]
            cube[:, :, sample] = padded_data[idx0::tile_x, idx1::tile_y]

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(data=cube, name='3D_CUBE')
    ])

    for idx0 in range(cube_size):
        hdul.append(fits.ImageHDU(
            data = cube[:, :, idx0],
            header = fits.Header({'WAVELEN': wl[idx0]})
        ))
    hdul.writeto('cubed_image.fits', overwrite=True)
    hdul.close()


#################################################################################################################
#################################################################################################################

## takes an multispectral cube and combines it into the image
## file_path - path to the .fits image file
## fits_path_sensor_layout - 
## tile_x, tile_y - side sizes of a wavelengths matrix
## return ndarray or the reconstructed image


def decube(fits_path_image, sensor_layout, tile_x, tile_y):
    with fits.open(fits_path_image) as hdul:
        num_bands = len(hdul) - 1
        cube_image_refl = np.stack([
            hdul[idx].data for idx in range(1, num_bands + 1)
        ], axis=-1)

        cube_image_wl = np.array([
            hdul[idx].header['WAVELEN'] for idx in range(1, num_bands + 1)
        ])
        cube_image_header = hdul[0].header

    if isinstance(sensor_layout, np.ndarray):
        sensor_wl = sensor_layout
    else:
        with fits.open(sensor_layout) as hdul:
            sensor_wl = hdul[0].data
        
    size_x, size_y, _ = cube_image_refl.shape

    band_pos = np.zeros((tile_x, tile_y), dtype=int)
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            band_pos[idx0, idx1] = np.argmin(np.abs(cube_image_wl - sensor_wl[idx0, idx1]))

    image_recon = np.zeros((size_x, size_y), dtype=float)
    for idx0 in range(tile_x):
        for idx1 in range(tile_y):
            band_idx = band_pos[idx0, idx1]
            image_recon[idx0::tile_x, idx1::tile_y] = cube_image_refl[idx0::tile_x, idx1::tile_y, band_idx]
    
    hdul = fits.HDUList([
        fits.PrimaryHDU(
            data = image_recon, header = cube_image_header
        )
    ])
    
    hdul.append(fits.ImageHDU(
        data = image_recon,
        header = cube_image_header
    ))
    
    hdul.writeto('decubed_image.fits', overwrite=True)
    hdul.close()