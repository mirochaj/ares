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

# Set chemistry - hydrogen only for this test
Z = [1]

# Set column density interval of interest (and number of sampling points)
logN = [np.linspace(15, 20, 51)]

# Set source properties - grab 10^5 K blackbody from RT06 #2
src = {'problem_type': 2}

# Initialize optimization object - use simulated annealing rather than MCMC
sedop = ares.inference.SpectrumOptimization(logN=logN, Z=Z, nfreq=1, 
    rs=src, mcmc=False, isothermal=False, thinlimit=True)

# Run optimization
sedop(1e4, burn=1e3, guess=[30., 1.0], err=0.01, step=[5, 0.05], 
    afreq=10, gamma=0.99)

# Compute cost function by brute force
E = np.linspace(15, 50, 100)
LE = np.linspace(0.6, 1.0, 100)
lnL = np.zeros([len(E), len(LE)])
for i in xrange(len(E)):
    for j in xrange(len(LE)):
        lnL[i,j] = sedop.cost(np.array([E[i], LE[j]]))

# Plot contours of cost function, analytic solution, and last 1000 steps of chain
pl.contour(E, LE, lnL.T, 3, colors='k', linestyles=[':', '--', '-'])
pl.plot(sedop.sampler.chain, sedop.sampler.chain, color='b')
    
# We should recover the mean ionizing photon energy and the 
# fraction of the bolometric luminosity emitted above 13.6 eV
Emono = sedop.rs.hnu_bar[0]
Lmono = sedop.rs.fLbol_ionizing[0]

# Plot what we expect    
pl.scatter([Emono]*2, [Lmono]*2, color='k', marker='+', s=200)

# Nice labels
pl.xlabel(r'$h\nu \ (\mathrm{eV})$')
pl.ylabel(r'$L_{\nu}$')
pl.xlim()


raw_input('click <enter> for confidence regions (E, LE)')
pl.close()

# Histogram MCMC results and plot
Ebins = np.linspace(Emono-5, Emono+5, 51)
Lbins = np.linspace(Lmono-0.1, Lmono+0.1, 51)
histE, binsE = np.histogram(sedop.sampler.chain[...,0], bins = Ebins)
histLE, binsLE = np.histogram(sedop.sampler.chain[...,1], bins = Lbins)

# Grab multiplot class
mp = MultiPanel(dims=(1,2))

mp.grid[0].plot(rebin(Ebins), histE, color = 'k', 
    drawstyle = 'steps-mid')
mp.grid[1].plot(rebin(Lbins), histLE, color = 'k', 
    drawstyle = 'steps-mid')

mp.grid[0].plot([Emono] * 2, mp.grid[0].get_ylim(), color = 'k', ls = ':')
mp.grid[1].plot([Lmono] * 2, mp.grid[1].get_ylim(), color = 'k', ls = ':')

mp.grid[0].set_xlabel(r'$h\nu \ (\mathrm{eV})$')
mp.grid[1].set_xlabel(r'$L_{\nu}$')
mp.grid[0].set_ylabel('Number of Samples')

mp.fix_ticks()
pl.draw()
raw_input('click <enter> for Phi and Psi')
pl.close()

# Plot optimal Phi and Psi functions vs. HI column density
pars = sedop.sampler.xarr_ML
Eopt, LEopt = np.array(pars[:len(pars) / 2]), np.array(pars[len(pars) / 2:])

best_phi = sedop.discrete_tabs(Eopt, LEopt)['logPhi_h_1']
best_psi = sedop.discrete_tabs(Eopt, LEopt)['logPsi_h_1']

mp = MultiPanel(dims=(2,1))

mp.grid[0].loglog(10**sedop.logN[0], 10**sedop.rs.tabs['logPhi_h_1'], color='k')
mp.grid[0].loglog(10**sedop.logN[0], 10**best_phi, color='b')
mp.grid[0].set_ylim(1e6, 1e11)
mp.grid[0].set_ylabel(r'$\Phi$')
mp.grid[0].set_xlabel(r'$N \ (\mathrm{cm}^{-2})$')

mp.grid[1].loglog(10**sedop.logN[0], 10**sedop.rs.tabs['logPsi_h_1'], color='k')
mp.grid[1].loglog(10**sedop.logN[0], 10**best_psi, color='b')
mp.grid[1].set_ylim(1e-4, 2)
mp.grid[1].set_ylabel(r'$\Psi$')

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
