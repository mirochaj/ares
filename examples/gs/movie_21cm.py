"""

make_movie.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Dec  9 17:10:52 MST 2014

Description: Make a sequence of images of the signal wiggling around as a 
function of some parameters.

Note: Can run this in parallel.

"""

import numpy as np
import ares, pickle
import matplotlib.pyplot as pl

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

parvals_lo = np.logspace(-1, 0, 11)
parvals_hi = np.logspace(0, 1., 11)

base_pars = {'verbose': False, 'progress_bar': False, 'final_redshift': 6.}
refvals = {'Nlw': 1e4, 'Nion': 4e3, 'fX': 0.2, 'fstar': 0.1}

# Loop over parameters of interest
frame = 0
for par in ['Nlw', 'fX', 'Nion', 'fstar']:

    allvals = np.concatenate((parvals_lo[-1::-1], parvals_lo,
                              parvals_hi, parvals_hi[-1::-1]))

    for val in allvals:

        if frame % size != rank:
            frame += 1
            continue

        kwargs = refvals.copy()
        kwargs.update(base_pars)
        kwargs[par] = refvals[par] * val

        sim = ares.simulations.Global21cm(**kwargs)
        sim.run()

        anl = ares.analysis.Global21cm(sim)
        ax = anl.GlobalSignature(ymax=60, ymin=-200)
        ax.set_ylim(-180, 60)

        ax.annotate(r'$f_{\alpha} = %.2g$' % (sim.pf['Nlw'] / refvals['Nlw']), 
            (15, 40), ha='left')
        ax.annotate(r'$f_{X} = %.2g$' % (sim.pf['fX'] / refvals['fX']), 
            (55, 40), ha='left')        
        ax.annotate(r'$f_{\mathrm{ion}} = %.2g$' % (sim.pf['Nion'] / refvals['Nion']),
            (95, 40), ha='left')    
        ax.annotate(r'$f_{\ast} = %.2g$' % sim.pf['fstar'],
            (135, 40), ha='left')        

        pl.draw()
        pl.savefig('frame_%s.png' % (str(frame).zfill(5)), dpi=120)
        
        ax.clear()

        frame += 1

    