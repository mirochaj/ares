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
## compute reionization history: compare to Figure 3
#
      
pars = \
{
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
 'final_redshift': 4.,
 'include_igm': False,  # single-zone model
 'cgm_Tk': 2e4,         # should be 20,000 K
 'clumping_factor': 3.,
 'pop_fesc': 0.2,
 'cgm_recombination': 'B',
}

fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
for T in np.arange(1, 2.1, 0.1):
    pars['cgm_Tk'] = T * 1e4
    
    sim = ares.simulations.MultiPhaseMedium(**pars)
    
    sim.run()

    ax2.plot(sim.history['z'], 1. - sim.history['cgm_h_2'])
    
ax2.set_xlim(4.5, 12)
ax2.set_ylim(0, 1.0)
ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'$1 - Q_{\mathrm{HII}}$')
      
pl.show()
