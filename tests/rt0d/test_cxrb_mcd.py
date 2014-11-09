"""

test_cxrb_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 19 12:40:52 2012

Description: Compare CXRB at z=6 for MCD and PL spectra.
"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# 
## INPUT
zf, zi = (6, 10)
E = np.logspace(2, 4.5, 100)
##
#

# Initialize radiation background
plpars = \
{
 'source_type': 'bh',
 'model': -1,
 'spectrum_type': 'pl',
 'spectrum_alpha': -1.5,
 'spectrum_Emin': 1e2,
 'spectrum_Emax': 3e4,
 'spectrum_EminNorm': 5e2,
 'spectrum_EmaxNorm': 8e3,
 'spectrum_logN': 21,
 'approx_xray': 0,
 'approx_helium': 0,
 'norm_by': 'xray',
}

plsrc = ares.solvers.UniformBackground(**plpars)

mcdpars = \
{
 'source_type': 'bh',
 'model': -1, 
 'source_mass': 10,
 'source_rmax': 1e3,
 'spectrum_type': 'mcd',
 'spectrum_Emin': 1e2,
 'spectrum_Emax': 3e4,
 'spectrum_EminNorm': 0,
 'spectrum_EmaxNorm': np.inf,
 'norm_by': 'xray',
 'approx_xray': 0,
 'approx_lya': 0,
}

mcdsrc = ares.solvers.UniformBackground(**mcdpars)

# Loop over sources and plot CXRB
colors = ['k', 'b']
for i, src in enumerate([plsrc, mcdsrc]):
    
    flux_neu = np.zeros_like(E)
    flux_ion = np.zeros_like(E)
    
    for j, nrg in enumerate(E): 
        flux_neu[j] = src.AngleAveragedFlux(6., nrg, zf=10, energy_units=True,
            xavg=lambda z: 0.0)
        flux_ion[j] = src.AngleAveragedFlux(6., nrg, zf=10, energy_units=True,
            xavg=lambda z: 1.0)

    if i == 0:
        label1, label2 = r'$x_i(z) = 1.0$', r'$x_i(z) = 0.0$'
    else:
        label1 = label2 = None
        
    # Plot up limiting cases with const. ionized fraction
    pl.loglog(E, flux_ion, color=colors[i], ls='-',
        label=label1)
    pl.loglog(E, flux_neu, color=colors[i], ls='--',
        label=label2)

pl.legend(frameon=False, loc='lower right')

# Make pretty                    
pl.xscale('log')
pl.yscale('log')
pl.xlim(min(E), max(E))
pl.xlabel(r'$h\nu \ (\mathrm{eV})$')
pl.ylabel(r'$J_{\nu} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$')    
pl.title(r'CXRB Spectrum at $z = 6$')
pl.xlim(1e2, 1e4)
pl.ylim(1e-30, 1e-23)
pl.annotate('PL', (0.85, 0.9), xycoords='axes fraction', color='k')
pl.annotate('MCD', (0.85, 0.8), xycoords='axes fraction', color='b')

    