import numpy as np

def correct_bckgr(raw_img : np.array, bckgr_im):
    """
    correct an image by a given background image. 
    The size of both images must be the same.

    Parameters
    ----------
    raw_img
    """

    clean_img = raw_img-bckgr_im
    return(clean_img)
