import numpy as np
from skimage.registration import phase_cross_correlation

def find_center(image, mask=None):
    """
    Find the center of an "image" with the
    "masked normalized cross-correlation" method introduced by Dirk Padfield.
    Mask from pyFAI must be inverted.
    :param image: diffraction image
    :param mask: mask, that masks beamstop, etc. inverted to the pyFAI-mask
    :type image: np.array
    :type mask: np.array
    :returns: x_position, y_position f the center.
    """
    if mask is None:
        mask = np.zeros(image.shape).astype("bool")
    mask = ~mask

    image_1 = image
    image_2 = np.flipud(np.fliplr(image))
    moving_mask = mask
    reference_mask = np.flipud(np.fliplr(mask))

    detected_shift = phase_cross_correlation(
        image_2, image_1, reference_mask=reference_mask, moving_mask=moving_mask
    )
    c_y = np.floor(image.shape[1] / 2 - 1 - detected_shift[0] / 2 + 0.5)
    c_x = np.floor(image.shape[0] / 2 - 1 - detected_shift[1] / 2 + 0.5)
    return c_x, c_y