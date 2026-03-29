import numpy as np
from scipy.ndimage import median_filter

def hotpixel_filter(data, tolerance=3, size=10):
    """
    Reduce the noise in the given 2D dataset.
    Returns the positions of outliers and the corrected image.

    Implemented methods for outlier detection: "mad_local": Median absolute deviation of nearest neighbors.
    """
    # The data type is changed from an unsigned int (original images) to float64 for further calculation
    if data.dtype != "float64":
        data = np.array(data, dtype="float64")

    blurred = median_filter(data, size=size)
    outliers = find_outlier_pixels_mad_local(
                data, blurred, tolerance, size
            )

    fixed_image = np.copy(data)  # This is the image with the hot pixels removed
    for y, x in zip(outliers[0], outliers[1]):
        fixed_image[y, x] = blurred[y, x]

    return outliers, fixed_image

def find_outlier_pixels_mad_local(data, blurred, tolerance, size):
    """Find outliers with the median absolut deviation (MAD)"""
    difference = np.abs(data - blurred)

    # Allow the difference of a pixel and its local median
    # to be `tolerance` times larger than the local median of differences.
    MAD = median_filter(difference, size=size)
    k = 1.4826  # from https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
    threshold = tolerance * MAD * k

    # find the hot pixels
    outliers = np.nonzero(difference > threshold)
    return outliers