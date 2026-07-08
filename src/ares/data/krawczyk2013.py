"""

krawczyk2013.py

"""

from . import ARES
from astropy.io import ascii

data = ascii.read(f"{ARES}/krawczyk_sed/apjs468686t2_mrt.txt")

xunit = 'hz'

x = 10**data['lognu']
y = 10**data['All']

# Correct bolometric normalization
#Lbol_template = np.trapz(Lnu_rest * nu_rest, x=np.log(nu_rest))

#def get_spectrum(E, t=0.0, **kwargs):
#    """
#    Broadband quasar template spectrum.
