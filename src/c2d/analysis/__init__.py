from c2d.analysis.error import (
    generalised_gaussian_fitting,
    laplacian_fitting,
    normal_fitting,
)
from c2d.analysis.interp import (
    aproximate_distribution,
    extract_max_min_wl_value,
    extract_wl_median,
    find_coldPixels,
    find_hotPixels,
    med_per_band,
    neighbours_compare,
    pip1,
    plot_med_interVSnotInterp,
    print_corrleation,
    reduce_interp,
    square_pattern,
    star_pattern,
)
from c2d.analysis.metrics import macropixel_error, rmse
from c2d.analysis.section import pipeline
from c2d.analysis.spectra import (
    all_spectra_mean,
    all_spectra_median,
    extract_dataset,
    extract_sample,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # error fitting
    "laplacian_fitting",
    "generalised_gaussian_fitting",
    "normal_fitting",
    # interpolation & filtering
    "reduce_interp",
    "extract_wl_median",
    "extract_max_min_wl_value",
    "star_pattern",
    "square_pattern",
    "find_coldPixels",
    "find_hotPixels",
    "neighbours_compare",
    "plot_med_interVSnotInterp",
    "print_corrleation",
    "pip1",
    "med_per_band",
    "aproximate_distribution",
    # metrics
    "rmse",
    "macropixel_error",
    # pipeline
    "pipeline",
    # spectra
    "extract_dataset",
    "extract_sample",
    "all_spectra_median",
    "all_spectra_mean",
]