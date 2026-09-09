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

    E_kin = voltage * e
    gamma = 1+ (E_kin/(m_e*c**2))
    wavelength = h/(m_e * c * gamma * np.sqrt(1-(1/gamma**2))) 

    return(wavelength)


