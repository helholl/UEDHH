from skued import biexponential, with_irf
import numpy as np

############################################
# fit function for transients
def biexp_irf_fit(time, *args, **kwargs):
    """
    fit function for biexponendtial transients. Use skued biexp. fit

    args: tuple 
        (time_resolution, t0_fit, amp1, amp2, tconst1, tconst2, offset)

    """
    args = list(args)
    args = tuple(args)
    return with_irf(np.abs(args[0]))(biexponential)(time, *args[1:], **kwargs)
############################################