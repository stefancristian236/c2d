"""
top-level package initialization.
"""

__version__ = "0.1.1"

from c2d import analysis, core, io, utils

from c2d.core import Cutout2D, cube, decube, get_tile_wavelengths, cut_cube, round_trip_check, spectra_pipeline

from c2d.analysis import pipeline, rmse

from c2d.io import extract_parameters

from c2d.utils import make_plot, make_grid 

__all__ = [
    # Metadata
    "__version__",
    
    # Subpackages
    "analysis",
    "core",
    "io",
    "utils",
    
    # Core API
    "Cutout2D",
    "cube",
    "decube",
    "get_tile_wavelengths",
    "cut_cube",
    'round_trip_check',
    'spectra_pipeline',
    
    # Analysis API
    "pipeline",
    "rmse",
    
    # IO API
    "extract_parameters",
    
    # Utils API
    "plotting",
    'make_plot',
    'make_grid',
]