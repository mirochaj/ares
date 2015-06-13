"""

test_generator_xrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description:

"""

import numpy as np
import os, sys, ares
import matplotlib.pyplot as pl

zi, zf = (40., 10.)

# Initialize radiation background
src_pars = \
{
 # Source properties
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.01 / (1. + z)**3.,
 'pop_sed': 'pl',
 'pop_alpha': 1.0,
 'pop_Emin': 8.,
 'pop_Emax': 500.,
 'pop_EminNorm': 5e2,
 'pop_EmaxNorm': 8e3,
 'pop_yield': 2.6e39,
 'pop_yield_units': 'erg/s/SFR',
 
 # Solution method
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 'include_H_Lya': False,
 
 'sawtooth_nmax': 8,
 
 'initial_redshift': zi,
 'final_redshift': zf,
}

rad1 = ares.simulations.MetaGalacticBackground(pop_sawtooth=True, **src_pars)
rad2 = ares.simulations.MetaGalacticBackground(pop_sawtooth=False, **src_pars)

"""
First, look at background flux itself.
"""

# Compute background flux w/ generator
rad1.run()
rad2.run()

z1, E1, flux1 = rad1.get_history(flatten=True)
z2, E2, flux2 = rad2.get_history(flatten=True)

pl.loglog(E1, flux1[-1])
pl.loglog(E2, flux2[-1], ls='--')

# OK, so E1 is a list of lists, one per sub-band
#for i, chunk in enumerate(E1):
#    if type(chunk) is list:
#        pl.loglog(np.concatenate(E1[i]), flux1[-1][i])
#    else:
#        pl.loglog(E1[i], flux1[-1][i])
        
# Compute background flux using more sophisticated (and expensive!) integrator
#Enum = np.logspace(np.log10(2e2), 4.5, 100)
#flux_num = np.array(map(lambda EE: rad.AngleAveragedFlux(zf, EE, zf=zi, 
#    h_2=lambda z: 0.0), Enum))

#fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)    
##ax1.semilogx(Enum, flux_num, color='k')
#ax1.loglog(E1[0], flux1[-1], color='k', ls='-')
##ax1.loglog(E2, flux2[-1], color='k', ls='--')
#ax1.set_xlabel(ares.util.labels['E'])
#ax1.set_ylabel(ares.util.labels['flux_E'])
#
#ax1.set_ylim(0.5 * flux_num[flux_num > 0].min(), 1.1 * flux_num.max())
  
"""
Next, look at ionization and heating rates.
"""

#fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
#fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)
#
#heat = []
#ioniz_rate = []
#ioniz_rate2 = []
#for i, redshift in enumerate(z):
#    heat.append(rad.volume.HeatingRate(z[i], xray_flux=fluxes[i,:]))    
#    ioniz_rate.append(rad.volume.IonizationRateIGM(z[i], xray_flux=fluxes[i,:]))
#    ioniz_rate2.append(rad.volume.SecondaryIonizationRateIGM(z[i], xray_flux=fluxes[i,:]))
#
#ax2.plot(z, heat)
#ax3.plot(z, ioniz_rate, label=r'$\Gamma_{\mathrm{HI}}$')
#ax3.plot(z, ioniz_rate2, label=r'$\gamma_{\mathrm{HI}}$')
#ax2.set_xlabel(r'$z$')
#ax2.set_ylabel(r'$\epsilon_X \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-3})$')
#ax3.set_xlabel(r'$z$')
#ax3.set_ylabel(r'Ionization Rate $(\mathrm{s}^{-1})$')
#ax3.legend(loc='best')
#pl.draw()
#
#