"""

test_cxrb_pl.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 19 12:40:52 2012

Description: Compare ionization and heating of background sourced by 
increasingly absorbed power-law X-ray emission.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Unabsorbed power-law
plpars = \
{
 'source_type': 'bh',
 'model': -1,
 'spectrum_type': 'pl',
 'spectrum_alpha': -1.5,
 'spectrum_Emin': 2e2,
 'spectrum_Emax': 3e4,
 'spectrum_EminNorm': 2e2,
 'spectrum_EmaxNorm': 3e4,
 'spectrum_logN': -np.inf,
 'approx_xrb': 0,
 'discrete_xrb': True,
 'redshift_bins': 400,
 'norm_by': 'xray',
}

plsrc = ares.solvers.UniformBackground(**plpars)

# Absorbed power-law
aplpars = plpars.copy()
aplpars.update({'spectrum_logN': 21.})

aplsrc = ares.solvers.UniformBackground(**aplpars)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)

# Loop over sources and plot CXRB
colors = ['k', 'b']
for i, rad in enumerate([plsrc, aplsrc]):
    
    rad.run()
    
    if np.isfinite(rad.pf['spectrum_logN']):
        label = r'$N = 10^{%i} \ \mathrm{cm}^{-2}$' % (rad.pf['spectrum_logN'])
    else:
        label = r'$N = 0 \ \mathrm{cm}^{-2}$'

    # Plot up background flux
    ax1.loglog(E, fluxes[0], color=colors[i], ls='-', label=label)
    
    # Plot up heating rate evolution
    heat = np.zeros_like(z)
    ioniz = np.zeros_like(z)
    for j, redshift in enumerate(z):
        heat[j] = rad.volume.HeatingRate(redshift, xray_flux=fluxes[j])
        ioniz[j] = rad.volume.IonizationRateIGM(redshift, xray_flux=fluxes[j])
    
    ax2.semilogy(z, heat, color=colors[i], ls='-', label=label)
    ax3.semilogy(z, ioniz, color=colors[i], ls='-', label=label)
    
    pl.draw()
    
# Make pretty                    
ax1.set_xlabel(ares.util.labels['E'])
ax1.set_ylabel(ares.util.labels['flux'])
ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'$\epsilon_X$')
ax3.set_xlabel(r'$z$')
ax3.set_ylabel(r'$\Gamma_{\mathrm{HI}}$')
ax1.set_ylim(1e-23, 1e-14)

for ax in [ax1, ax2, ax3]:
    ax.legend()

pl.draw()
    