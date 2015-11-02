"""

test_hmf_PS.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri May  3 16:03:57 2013

Description: Use Press-Schechter mass function to test numerical integration, 
since PS has an analytic solution for the collapsed fraction.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Two HMFs: one analytic, one numerical
hmf_a = ares.populations.HaloPopulation(hmf_func='PS', hmf_analytic=True)
hmf_n = ares.populations.HaloPopulation(hmf_func='PS', hmf_analytic=False,
    hmf_load=False)

"""
First, plot fcoll at a few redshifts, using symbols for numerical solutions.
"""

fig1 = pl.figure(1)
ax1 = fig1.add_subplot(111)

for i, z in enumerate(hmf_a.halos.z[120:-1:200]):
    ax1.loglog(hmf_a.halos.M[::25], hmf_a.halos.fcoll_tab[i][::25], 
        color='k')
    ax1.scatter(hmf_n.halos.M[::25], hmf_n.halos.fcoll_tab[i][::25], 
        color='b', marker='o', facecolors='none')

ax1.set_xlim(1e4, 1e12)
ax1.set_ylim(1e-3, 1)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$M / M_{\odot}$')
ax1.set_ylabel(r'$f_{\mathrm{coll}}$')

"""
Next, plot the error as a function of mass (and redshift).
"""

fig2 = pl.figure(2)
ax2 = fig2.add_subplot(111)

ls = ['-', '--', '-.', ':', '-']
for i, z in enumerate(hmf_a.halos.z[120:-1:200]):
    err1 = (hmf_n.halos.fcoll_tab[np.argmin(np.abs(z-hmf_n.halos.z))] - \
            hmf_a.halos.fcoll_tab[np.argmin(np.abs(z-hmf_a.halos.z))]) / \
            hmf_a.halos.fcoll_tab[np.argmin(np.abs(z-hmf_a.halos.z))]

    ax2.loglog(hmf_a.halos.M, np.abs(err1), color='b', ls=ls[i])
    ax2.loglog([hmf_a.halos.VirialMass(1e4, z, mu=1.22)]*2, [1e-8, 1],
        color='k', ls=ls[i])

ax2.annotate(r'$T_{\mathrm{min}}=10^4 \ \mathrm{K}$', (0.05, 0.9),
    xycoords='axes fraction')
ax2.set_xlim(1e4, 1e11)
ax2.set_ylim(1e-8, 1)
ax2.set_xlabel(r'$M / M_{\odot}$')
ax2.set_ylabel('relative error')
ax2.set_title(r'$z=10,20,30,40,50$')

"""
Now, plot the rate of collapse as a function of redshift.
"""

fig3 = pl.figure(3)
ax3 = fig3.add_subplot(111)

ls = '-', '--', ':', '-.'
z = np.linspace(10.1, 49.9, 500)
for i, T in enumerate([300., 1e3, 1e4, 1e5]):
    pop = ares.populations.GalaxyPopulation(hmf_func='PS', 
        hmf_analytic=True, hmf_load=False, pop_Tmin=T)

    f = np.array(map(pop.fcoll, z))
    dfdz = np.array(map(pop.dfcolldz, z))
    
    ax3.semilogy(z, dfdz, color='k', ls=ls[i], 
        label=r'$T_{\min} = 10^{%.2g} \ \mathrm{K}$' % np.log10(T))

ax3.set_xlabel(r'$z$')
ax3.set_ylabel(r'$df_{\mathrm{coll}} / dz$')
ax3.legend(loc='lower left', fontsize=12, ncol=2)

pl.draw()
