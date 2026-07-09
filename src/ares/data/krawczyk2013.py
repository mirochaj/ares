"""

krawczyk2013.py

"""

from . import ARES
import numpy as np
from astropy.io import ascii
from ..physics.Constants import h_p, erg_per_ev
from ..util.Misc import numeric_types
from ..util.Math import integrate_with_subgrid_interp
data = ascii.read(f"{ARES}/krawczyk_sed/apjs468686t2_mrt.txt")

xunit = 'hz'

# Save frequencies and energies
x = 10**data['lognu']
x_eV = h_p * x / erg_per_ev
# Convert from provided nu * L_nu back to L_nu
y = 10**data['All'] / x

norm = 8.7025e31

def get_spectrum(E, t=0.0, **kwargs):
    """
    Broadband quasar template spectrum, normalized such that it 
    integrates to unity (if integrating over x/[eV])
    """

    if type(E) in numeric_types:
        return np.interp(E, x_eV, y / norm, left=0, right=0)
    else:
        return integrate_with_subgrid_interp(x_eV, y / norm, E[0], E[1])

