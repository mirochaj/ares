"""

test_qso_pop.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May 26 14:32:20 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Simpler model for redshift evolution
# Just a piecewise power-law about z=2.5
zfunc = lambda z: ((1. + z) / 3.5)**2. if z < 2.5 else ((1. + z) / 3.5)**-2. 

# Initialize a GalaxyPopulation
pop = ares.populations.GalaxyPopulation(spectrum_Emin=2e2,
    lf_norm=1e-4, spectrum_Emax=8e3, source_sed='sazonov2004', lf_zfunc=zfunc,
    approx_xrb=False, source_type='bh')
    
#L = np.logspace(40, 44, 50)
#pl.loglog(L, phi_L1)
#pl.loglog(L, phi_L2)

z = np.arange(0, 6, 0.05)

Lx1 = map(lambda zz: pop.LuminosityDensity(zz, Lmax=1e44, Emin=5e2, Emax=2e3), z)
Lx2 = map(lambda zz: pop.LuminosityDensity(zz, Lmax=1e44, Emin=2e3, Emax=8e3), z)
pl.semilogy(z, Lx1)
pl.semilogy(z, Lx2)

