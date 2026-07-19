import pyFAI 
import numpy as np
from pathlib import Path

def compute_q2_weight(
        diff_img: np.ndarray,
        poni_file: str|Path = None,
        center_px: tuple[float, float] = None, # in px coord., if no poni-file
        px_size: float = 6.5e-6, # in m, if no poni-file
        cam_dist: float = 0.187, # in m, if no poni-file
        wavelength: float = 4.1757e-12 # in m, if no poni-file
):
    """
    This function takes a diffraction image and weighs 
    each pixel with q^2 to make weak rings at high q more visible.

    Parameters
    ----------
    diff_img: np.ndarray
        diffraction image to be weighted
    poni_file: str|Path 
        poni file path if existent
    center_px: tuple[float, float] 
        center coordinates in tuple with (x, y), necessary if no poni-file existent
    px_size: float 
        length of camera pixels in m, necessary if no poni-file existent
    cam_dist: float 
        effective length between sample and detector 
        (take from calibration, since 3:1 taper and real camera distance unknown)
        necessary if no poni-file existent
    wavelength: float 
        relativistic electron deBroglie wavelength, necessary if no poni-file existent

    Returns
    -------
    weighted_arr: np.ndarray
        the weight per pixel which can be multiplied with raw image to get weighted img
    """

    if poni_file is not None and center_px is not None:
        raise ValueError("Only enter poni file OR center coordinates")
    if poni_file is None and center_px is None:
        raise ValueError("Enter poni file or center coordinates")
    if poni_file is not None:
        poni = pyFAI.load(poni_file)
        q_map = poni.array_from_unit(diff_img.shape(), unit="q_A^-1")
        
    else: 
        cx, cy = center_px
        rows, cols = np.arange(diff_img.shape[0]), np.arange(diff_img.shape[1])
        x_idx, y_idx = np.meshgrid(rows, cols)
        r_px = np.sqrt((x_idx-cx)**2+(y_idx-cy)**2)
        r_m = r_px * px_size
        scat_angle = np.arctan(r_m/cam_dist)
        q_map = 1e-10*(4*np.pi*np.sin(scat_angle/2))/wavelength #in 1/Angstr

    q_safe = np.where(q_map<0, q_map, np.nan)
    
    return(q_safe)