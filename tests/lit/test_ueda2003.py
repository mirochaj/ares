"""

test_qsolf.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri May 29 20:29:54 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Create a Ueda et al. 2003 module
u03 = ares.util.read_lit('ueda2003')

z = np.arange(0, 3)
L = np.logspace(41, 46)

"""
Plot the QSO-LF at a few redshifts, show PLE vs. PDE models.
"""
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

colors = 'k', 'b', 'g'
for i, redshift in enumerate(z):
    lf_ple = u03.LuminosityFunction(L, redshift, evolution='ple')
    lf_pde = u03.LuminosityFunction(L, redshift, evolution='pde')
    
    ax1.loglog(L, lf_ple, color=colors[i], ls='--')
    ax1.loglog(L, lf_pde, color=colors[i], ls='-', label=r'$z=%i$' % redshift)
    
ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')
pl.show()

"""
Plot z=0 PDE best-fit QSO-LF w/ blue semi-transparent lines as random samples from
the posterior PDF.
"""
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

pop = ares.analysis.Population(u03)

LFz0 = lambda LL, **kwargs: u03.LuminosityFunction(LL, z=0.0, **kwargs)
models = pop.SamplePosterior(L, LFz0, u03.qsolf_pde_pars, u03.qsolf_pde_err)

for i in range(int(models.shape[1])):
    ax2.loglog(L, models[:,i], color='b', alpha=0.05)

ax2.loglog(L, u03.LuminosityFunction(L, 0.0, evolution='pde'))    
ax2.set_xlabel(r'$L_X$')    
ax2.set_ylabel(r'$\phi(L_X)$')
    
pl.show()

"""
Convert to another band assuming some SED.
"""

#fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)
#pop = ares.populations.GalaxyPopulation(source_type='bh', approx_xrb=False, 
#    source_sed='sazonov2004', pop_lf='ueda2003')
#
#pl.loglog(L, map(lambda LL: pop.LuminosityFunction(LL, z=0, Emin=2e3, Emax=1e4), L), 
#    color='k', label=r'$2-10 \ \mathrm{keV}$')
#pl.loglog(L, map(lambda LL: pop.LuminosityFunction(LL, z=0, Emin=5e2, Emax=2e3), L), 
#    color='b', label=r'$0.5-2 \ \mathrm{keV}$')
#    
#ax3.set_xlabel(r'$L_X$')          
#ax3.set_ylabel(r'$\phi(L_X)$')
#    
#pl.legend(loc='best')
    


