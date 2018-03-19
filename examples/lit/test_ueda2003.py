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
Plot the QSO-LF at a few redshifts, show PLE, PDE, LDDE models.
"""
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

colors = 'k', 'b', 'g'
for i, redshift in enumerate(z):
    lf_ple = u03.LuminosityFunction(L, redshift, evolution='ple')
    lf_pde = u03.LuminosityFunction(L, redshift, evolution='pde')
    lf_ldde = u03.LuminosityFunction(L, redshift, evolution='ldde')
    
    ax1.loglog(L, lf_ple, color=colors[i], ls='--')
    ax1.loglog(L, lf_pde, color=colors[i], ls='-', label=r'$z={}$'.format(redshift))
    ax1.loglog(L, lf_ldde, color=colors[i], ls=':')
    
ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')
pl.show()

"""
Plot the different evolution functions.
"""
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

z = np.linspace(0, 5, 100)

zc_pde = lambda z: u03._evolution_factor_pde(z, **u03.qsolf_pde_pars)
zc_ldde = lambda z, LL: u03._evolution_factor_ldde(z, LL, **u03.qsolf_ldde_pars)

ax2.plot(z, list(map(zc_pde, z)))

colors = ['b', 'r', 'g']
for i, interval in enumerate([(41, 43), (43, 44), (44, 45)]):

    f1 = lambda zz: zc_ldde(zz, 10**interval[0])
    f2 = lambda zz: zc_ldde(zz, 10**interval[1])
    ax2.fill_between(z, list(map(f1, z)), list(map(f2, z)), color=colors[i])
    ax2.plot(z, list(map(f1, z)), color=colors[i], 
        label=r'$10^{{{0}}} \leq L_X \leq 10^{{{1}}}$'.format(*interval))
    
ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'$e(z)$')
ax2.legend(fontsize=16)
pl.show()

"""
Plot the cutoff redshift vs. L_X. Figure 10 in paper.
"""

fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)

L = np.logspace(41, 47)
zc = lambda LL: u03._zc_of_L(LL, **u03.qsolf_ldde_pars)
ax3.semilogx(L, list(map(zc, L)))
ax3.set_xlabel(r'$L_X$')
ax3.set_ylabel(r'$z_c$')
pl.show()

"""
Plot z=0 PDE best-fit QSO-LF w/ blue semi-transparent lines as random samples from
the posterior PDF.
"""
fig4 = pl.figure(4); ax4 = fig4.add_subplot(111)

pop = ares.analysis.Population(u03)

for j, evolution in enumerate(['pde', 'ple', 'ldde']):
    LFz0 = lambda LL, **kwargs: u03.LuminosityFunction(LL, z=0.0, 
        evolution=evolution, **kwargs)
    
    models = pop.SamplePosterior(L, LFz0, u03.kwargs_by_evolution[evolution], 
        u03.errs_by_evolution[evolution], Ns=250)
    
    for i in range(int(models.shape[1])):
        ax4.loglog(L, models[:,i], color=colors[j], alpha=0.05)
    
    ax4.loglog(L, u03.LuminosityFunction(L, 0.0, evolution=evolution))    

ax4.set_xlabel(r'$L_X$')    
ax4.set_ylabel(r'$\phi(L_X)$')
    
pl.show()



