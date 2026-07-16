"""

krawczyk2013.py

"""

import warnings
from . import ARES
import numpy as np
from astropy.io import ascii
from astropy.units import UnitsWarning
from ..physics.Constants import h_p, erg_per_ev
from ..util.Misc import numeric_types
from ..util.Math import integrate_with_subgrid_interp

warnings.simplefilter('ignore', UnitsWarning)

data = ascii.read(f"{ARES}/krawczyk_sed/apjs468686t2_mrt.txt")

xunit = 'hz'

# Save frequencies and energies
x = np.array(10**data['lognu'])
x_eV = h_p * x / erg_per_ev
# Convert from provided nu * L_nu back to L_nu
y = np.array(10**data['All'] / x)

norm = 8.7025e31

Emin = np.min(x_eV)
Emax = np.max(x_eV)

def get_spectrum(E, t=0.0, **kwargs):
    """
    Broadband quasar template spectrum, normalized such that it 
    integrates to unity (if integrating over x/[eV])
    """

    if (type(E) in numeric_types) or (len(E) == 1):
        return np.interp(E, x_eV, y / norm, left=0, right=0)
    else:
        return integrate_with_subgrid_interp(x_eV, y / norm, 
            max(min(E), Emin), min(max(E), Emax))

