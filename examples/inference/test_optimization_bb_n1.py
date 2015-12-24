"""

test_optimization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 25 09:27:17 2013

Description: Optimize 10^5 BB SED in optically thin limit, in which case
we have an ~analytic solution that we can compare to.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.util.Stats import rebin
from ares.analysis import MultiPanel
from ares.physics.Constants import erg_per_ev 

# Set source properties - grab 10^5 K blackbody from RT06 #2
pars = \
{
 'problem_type': 2,
 'tables_logN': [np.linspace(15, 20, 51)],
 'isothermal': True,
 'secondary_ionization': 0,
}

# Initialize optimization object - use simulated annealing rather than MCMC
sedop = ares.inference.SpectrumOptimization(**pars)

sedop.nfreq = 1
sedop.guess = [30.,1.0]
sedop.thinlimit = True

# Run optimization
sedop.run(niter=1e3)

# Compute cost function by brute force
E = np.linspace(15, 50, 100)
LE = np.linspace(0.6, 1.0, 100)
lnL = np.zeros([len(E), len(LE)])
for i in xrange(len(E)):
    for j in xrange(len(LE)):
        lnL[i,j] = sedop.cost(np.array([E[i], LE[j]]))

# Plot contours of cost function, analytic solution, optimal solution
pl.contour(E, LE, lnL.T, 3, colors='k', linestyles=[':', '--', '-'])
pl.scatter(*sedop.pars, color='b', marker='o', s=100, 
    facecolors='none')

# We should recover the mean ionizing photon energy and the 
# fraction of the bolometric luminosity emitted above 13.6 eV
Emono = sedop.rs.hnu_bar[0]
Lmono = sedop.rs.fLbol_ionizing[0]

# Plot what we expect    
pl.scatter([Emono]*2, [Lmono]*2, color='k', marker='+', s=200)

# Nice labels
pl.xlabel(r'$h\nu \ (\mathrm{eV})$')
pl.ylabel(r'$L_{\nu} / L_{\mathrm{bol}}$')

# Plot optimal Phi and Psi functions vs. HI column density
Eopt, LEopt = sedop.sampler.x

mp = MultiPanel(dims=(2,1), fig=2)

for i, integ in enumerate(sedop.integrals):
    integral = 'log%s_h_1' % integ
    best_int = sedop.discrete_tabs(Eopt, LEopt)[integral]
    mp.grid[i].loglog(10**sedop.logN[0], 10**sedop.rs.tabs[integral], color='k')
    mp.grid[i].loglog(10**sedop.logN[0], 10**best_int, color='b')

mp.grid[0].set_ylim(1e6, 1e11)
mp.grid[0].set_ylabel(r'$\Phi(N)$')
mp.grid[0].set_xlabel(r'$N \ (\mathrm{cm}^{-2})$')

mp.grid[1].set_ylim(1e-4, 2)
mp.grid[1].set_ylabel(r'$\Psi(N)$')

mp.fix_ticks()

raw_input('click <enter> to proceed with verification of solution')
pl.close()

print 'Verifying optimal discrete SED with ARES...\n'

"""
Run ARES to see how this solution works.
"""

continuous = ares.simulations.RaySegment(pf = {'problem_type': 2, 'grid_cells': 32})
continuous.run()

discrete = ares.simulations.RaySegment(pf = {'problem_type': 2.1, 'grid_cells': 32, 
    'spectrum_E': Eopt, 'spectrum_LE': LEopt})
discrete.run()

c = ares.analysis.RaySegment(continuous.checkpoints)    
d = ares.analysis.RaySegment(discrete.checkpoints)

ax = c.IonizationProfile(t=[10, 30, 100])
d.IonizationProfile(t=[10, 30, 100], color='b', ax=ax)

raw_input('click <enter> for temperature profiles')
pl.close()

ax = c.TemperatureProfile(t=[10, 30, 100])
d.TemperatureProfile(t=[10, 30, 100], color='b', ax=ax)

raw_input('\nSee! Monochromatic SEDs don\'t do so great.')
pl.close()
