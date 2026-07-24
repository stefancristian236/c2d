from xmlrpc.client import FastMarshaller

import libs.c2d.cube_decube as c2d
import libs.c2d.plot as plt2
import libs.c2d.spectra_calculator as spc
import libs.c2d.metric as mtr
import importlib
from astropy.io import fits
from astropy.nddata import Cutout2D
import numpy as np
import matplotlib.pyplot as plt
importlib.reload(c2d)
importlib.reload(plt2)
importlib.reload(spc)
importlib.reload(mtr)

def pipeline(data, center, size, title, cut_flag = False):
    if cut_flag == True:
        section = []
        for idx0 in range(data.shape[2]):
            section.append(Cutout2D(data[:, :, idx0], center, size).data)
        section = np.stack(section, axis = -1)
    else:
        section = data
    x, y, z, err = mtr.macropixel_error(section)
    
    print(x)
    print(y)
    print(z)

    im = plt.imshow(err,
                origin='lower',
                cmap='magma',
                vmin=np.min(err),
                vmax=np.max(err))

    plt.colorbar(im) 
    plt.savefig(f'{title}', dpi=1000)
    plt.show()
    
    return section, err