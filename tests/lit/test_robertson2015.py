"""

test_sfrd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 10:03:30 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Create a Robertson2015 module
r15 = ares.util.read_lit('robertson2015')

# Plot the best-fit SFRD model
z = np.arange(0, 15.05, 0.05)

# Plot 1,000 samples drawn from the Robertson et al. posteriors
pop = ares.analysis.Population(r15)
models = pop.SamplePosterior(z, r15.SFRD, r15.sfrd_pars, r15.sfrd_err)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
for i in range(int(models.shape[1])):
    ax1.plot(z, np.log10(models[:,i]), color='b', alpha=0.05)

# Plot best-fit (compare to their Figure 1)
ax1.plot(z, np.log10(r15.SFRD(z)))
ax1.set_xlim(0, 15)
ax1.set_ylim(-3.5, -0.4)
ax1.set_xlabel(r'$z$')
ax1.set_ylabel(ares.util.labels['sfrd'])
      
#
## compute reionization history: compare to their Figure 3
#
      
pars = \
{
 'problem_type': 100,
 'pop_type': 'galaxy',
 'pop_sfrd': 'robertson2015',
 'pop_sed': 'pl',
 'pop_alpha': 1.0,
 'pop_Emin': 13.6,
 'pop_Emax': 24.6,
 'pop_EminNorm': 13.6,
 'pop_EmaxNorm': 24.6,
 'pop_yield': 10**53.14,
 'pop_yield_units': 'photons/s/sfr',
 'initial_redshift': 30.,
 'final_redshift': 4.1,
 'include_igm': False,                   # single-zone model
 'cgm_initial_temperature': 2e4,         # should be 20,000 K
 'clumping_factor': 3.,
 'pop_fesc': 0.2,
 'cgm_recombination': 'B',
 'cgm_collisional_ionization': False,
}

fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)    
sim = ares.simulations.MultiPhaseMedium(**pars)
sim.run()

ax2.plot(sim.history['z'], 1. - sim.history['cgm_h_2'])

# By eye, where does EoR start and begin from R15
ax2.plot([6.2]*2, [0, 0.1], color='k', ls='--')    
ax2.plot([10,12], [0.93]*2, color='k', ls='--')    
    
ax2.set_xlim(4.5, 12)
ax2.set_ylim(0, 1.0)
ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'$1 - Q_{\mathrm{HII}}$')
      
anl = ares.analysis.MultiPhaseMedium(sim)
ax3 = anl.OpticalDepthHistory(fig=3, include_He=True, z_HeII_EoR=4., 
    label='ARES')

ax3.plot([0, 20], [0.066 - 0.012]*2, color='k', ls='--')
ax3.plot([0, 20], [0.066 + 0.012]*2, color='k', ls='--', 
    label=r'Planck $1-\sigma$')
ax3.plot([0, 20], [0.088 - 0.014]*2, color='k', ls=':')
ax3.plot([0, 20], [0.088 + 0.014]*2, color='k', ls=':', 
    label=r'WMAP $1-\sigma$')    
ax3.fill_between([0, 20], [0.066 - 0.012]*2, [0.066 + 0.012]*2, color='gray')
ax3.plot([16, 20], [0.06]*2, color='b', ls='-', label=r'R15')
ax3.legend(loc='lower right', fontsize=14)     
ax3.set_ylim(0, 0.11)
pl.show()
