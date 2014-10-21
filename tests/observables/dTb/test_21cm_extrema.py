"""

test_21cm_extrema.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May  6 18:10:46 MDT 2014

Description: Make sure our extrema-finding routines work.

"""

import ares
import matplotlib.pyplot as pl

blobs = (['z', 'dTb', 'igm_Tk', 'Ja'], ['B', 'C', 'D'])

sim = ares.simulations.Global21cm(track_extrema=True, inline_analysis=blobs)
sim.run()
                    
anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()

for TP in anl.turning_points:
    if TP == 'trans':
        continue

    z, T = anl.turning_points[TP][0:2]
    nu = ares.physics.Constants.nu_0_mhz / (1. + z)
    
    ax.scatter(nu, T, color='b', marker='x', s=150, lw=2)

pl.draw()


# Print out in-line analysis:
for tp in list('BCD'):
    sim.tabulate_blobs(tp)

