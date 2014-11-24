"""

test_sawtooth.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Aug 16 12:59:40 MDT 2013

Description: This is very similar to Haiman, Abel, & Rees (1997) Fig. 1.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import *

pars = \
{
 'source_type': 'star',
 'source_temperature': 1e5,
 'Tmin': 1e3,
 'fstar': 1e-2,
 'spectrum_type': 'bb',
 'spectrum_Emin': 5.,
 'spectrum_Emax': E_LL,
 'spectrum_EminNorm': 0.01,
 'spectrum_EmaxNorm': 5e2,
 'approx_He': 0,
 'approx_xray': 0,
 'approx_lya': 0,
 'Nlw': 9690.,
 'norm_by': 'lw',
 'is_ion_src_cgm': False,
 'is_ion_src_igm': False,
 'is_heat_src_igm': False,
 'initial_redshift': 40,
 'final_redshift': 10,
}
    
rad = ares.solvers.UniformBackground(**pars)
E = np.linspace(5.0, E_LL-0.01, 500)

color='k'
for zf in [15, 30]:
    F = map(lambda EE: rad.AngleAveragedFlux(zf, EE, xavg=lambda z: 0.0, 
        energy_units=True), E)
    Fthin = map(lambda EE: rad.AngleAveragedFlux(zf, EE, tau=0.0,
        energy_units=True), E)

    pl.semilogy(E, np.array(F) / 1e-21, color=color, label=r'$z=%i$' % zf)
    pl.semilogy(E, np.array(Fthin) / 1e-21, color=color, ls='--')
    
    #Ja = rad.LymanAlphaFlux(zf, energy_units=True)
    #pl.scatter(E_LyA, Ja / 1e-21, color=color, marker='+', s=150)
    
    color='b'

pl.xlabel(r'$h\nu \ (\mathrm{eV})$')
pl.ylabel(r'$J_{\nu} / J_{21}$')
pl.legend(loc='lower left', frameon=False)
pl.ylim(1e-1, 1e2)
