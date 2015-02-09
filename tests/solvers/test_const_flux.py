"""

test_const_ionization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Oct 16 14:46:48 MDT 2014

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.CrossSections import PhotoIonizationCrossSection as sigma

s_per_yr = ares.physics.Constants.s_per_yr

pars = \
{
 'problem_type': 0,
 'grid_cells': 1, 
 'initial_ionization': [1e-8],
 #'initial_temperature': 1e2,
 'isothermal': False,
 'source_type': 'toy',
 'stop_time': 10.0,
 'plane_parallel': True,
 'recombination': 0,
 'source_qdot': 1e10,
 'source_lifetime': 1e10,
 'spectrum_E': [13.61],
 'spectrum_LE': [1.0],
 'secondary_ionization': 0,
 'logdtDataDump': 0.5,
}

sim = ares.simulations.RaySegment(**pars)
sim.run()

anl = ares.analysis.RaySegment(sim.checkpoints)

t, xHI = anl.CellEvolution(field='h_1')

pl.loglog(t / s_per_yr, 1. - xHI, color='k', label='numerical')
pl.ylim(1e-8, 1.5)
pl.xlabel(r'$t / \mathrm{yr}$')
pl.ylabel(r'Ionized Fraction')

# Analytic solution
sigma0 = sigma(pars['spectrum_E'][0])
qdot = pars['source_qdot']
Gamma = qdot * sigma0

xi0 = 1e-8
C = 1. - xi0
def xi(t, Gamma=Gamma):
    return 1. - C * np.exp(-Gamma * t)

pars.update({'source_type': 'diffuse', 'Gamma': Gamma})

pl.scatter(t / s_per_yr, map(xi, t), color='b', facecolors='none', s=100,
    label='analytic')
pl.legend(loc='lower right')
