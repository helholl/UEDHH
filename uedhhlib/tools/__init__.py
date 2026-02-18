"""
Tools for UED data preprocessing and quality control.

These are standalone utilities that operate on directories/files,
not part of the core Dataset API.
"""

from .preprocessing import (
    remove_hot_pixels_from_dataset,
    #clean_specific_image_types,
    #verify_preprocessing
)

# from .datapicker import (
#     # Deine QC-Funktionen hier
#     detect_arcs,
#     plot_intensity_distribution,
#     # etc.
# )

__all__ = [
    # Preprocessing
    'remove_hot_pixels_from_dataset',
    #'clean_specific_image_types',
    #'verify_preprocessing',
    # Quality Control
    #'detect_arcs',
    #'plot_intensity_distribution',
]