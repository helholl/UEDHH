import numpy as np
from scipy.constants import h, m_e, c, e

def electron_wavelength_from_energy(voltage: int):
    """
    calculate relativistic electron wavelength from acceleration voltage 

    Parameters
    ----------
    voltage: int
        acceleration voltage in V

    Return
    ------
    wavelength: float

    """
    v=np.sqrt(2*voltage*e/m_e)
    wavelength = h/(m_e*v) * np.sqrt(1- (v/c)**2)

    return(wavelength)

print(electron_wavelength_from_energy(80000))