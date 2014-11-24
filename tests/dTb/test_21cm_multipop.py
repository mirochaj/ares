"""

test_21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Oct 1 15:23:53 2012

Description: Make sure the global 21-cm signal calculator works.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import rhodot_cgs

src1 = \
{
 'Tmin': 1e4,
 'source_type': 'star',
 'fstar': 1e-1,
 'Nion': 4e3,
 'Nlw': 9690.,
}

src2 = \
{
 'Tmin': 300.,
 'source_type': 'star',
 'fstar': 1e-3,
 'Nion': 3e4,
 'Nlw': 4800.,
}

pars = \
{
 'source_kwargs': [src1, src2],
}

# Dual-population model
sim = ares.simulations.Global21cm(**pars)
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature(color='k', label=r'dual-pop')

# Standard single population model - overplot
sim2 = ares.simulations.Global21cm()
sim2.run()

anl2 = ares.analysis.Global21cm(sim2)
ax = anl2.GlobalSignature(ax=ax, color='b', label='single-pop')

z = np.linspace(10, 40)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

pop1, pop2 = sim.pops.pops

ax2.semilogy(z, np.array(map(pop1.SFRD, z)) * rhodot_cgs, color='k', ls='-',    
    label='PopII')
ax2.semilogy(z, np.array(map(pop2.SFRD, z)) * rhodot_cgs, color='k', ls='--',
    label='PopIII')
ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'SFRD')
ax2.legend(loc='upper right')
pl.draw()