import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D

from c2d.analysis import metrics as mtr 

def pipeline(
    data: np.ndarray, 
    center: tuple, 
    size: tuple, 
    title: str, 
    cut_flag: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    processes 3D spectral data, extracts a cutout if requested, 
    calculates macropixel errors, and saves a visualization.
    """
    if cut_flag:
        section = np.stack(
            [Cutout2D(data[:, :, z], center, size).data for z in range(data.shape[2])], 
            axis=-1
        )
    else:
        section = data
        
    # calculate errors using your metrics module
    x, y, z, err = mtr.macropixel_error(section)
    
    print(f"X: {x}\nY: {y}\nZ: {z}")

    fig, ax = plt.subplots()
    im = ax.imshow(
        err,
        origin='lower',
        cmap='magma',
        vmin=np.min(err),
        vmax=np.max(err)
    )

    fig.colorbar(im, ax=ax) 
    fig.savefig(f"{title}", dpi=1000)
    plt.show()
    
    return section, err