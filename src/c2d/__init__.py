"""
top-level package initialization.
"""

from c2d import analysis, core, io, utils

from c2d.core import Cutout2D, cube, decube, get_tile_wavelengths, cut_cube, round_trip_check, spectra_pipeline

from c2d.analysis import pipeline, rmse, laplacian_fitting, generalised_gaussian_fitting, normal_fitting

from c2d.io import extract_parameters

from c2d.utils import make_plot, make_grid 

__all__ = [
    
    # subpackages
    "analysis",
    "core",
    "io",
    "utils",
    
    # core API
    "Cutout2D",
    "cube",
    "decube",
    "get_tile_wavelengths",
    "cut_cube",
    "round_trip_check",
    "spectra_pipeline",
    
    # analysis API
    "pipeline",
    "rmse",
    "laplacian_fitting",
    "generalised_gaussian_fitting",
    "normal_fitting",
    
    # io API
    "extract_parameters",
    
    # utils API
    "plotting",
    "make_plot",
    "make_grid",
]