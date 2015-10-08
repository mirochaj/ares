"""

test_21cm_cxrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 11 13:02:00 MST 2013

Description: 

"""

import os, ares
import matplotlib.pyplot as pl

pars = \
{
 'pop_sed{0}': 'bb',
 'pop_Emin{0}': 10.18,
 'pop_Emax{0}': 15.,
 'pop_EminNorm{0}': 10.2,
 'pop_EmaxNorm{0}': 13.6,
 'pop_yield{0}': 9690.,
 'pop_yield_units{0}': 'photons/baryon',
 'pop_solve_rte{0}': True,
 'pop_tau_Nz{0}': 1e3,
 'sawtooth_nmax': 13,
}

ax = None
ls = '-', '--'
colors = 'b', 'g', 'm'
for j, injected in enumerate([True, False]):
    tpB = []

    for i, logT in enumerate([4, 4.7, 5]):
        pars.update({'source_temperature{0}': 10**logT, 
            'lya_injected{0}': injected})
        sim2 = ares.simulations.Global21cm(**pars)
        sim2.run()

        if j == 0:
            label = r'$T_{\ast} = 10^{%.2g} \ \mathrm{K}$' % (logT)
        else:
            label = None

        anl2 = ares.analysis.Global21cm(sim2)
        ax = anl2.GlobalSignature(ax=ax, color=colors[i], label=label,  
            ls=ls[j])

# Fiducial model, approximate Lyman-alpha background
sim = ares.simulations.Global21cm()
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature(ax=ax, color='k', ls='-')

pl.draw()
