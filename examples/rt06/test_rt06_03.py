"""

test_rt06_03.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jan 18 09:35:35 2013

Description: Single zone ionization/recombination + heating/cooling test.

"""

import ares
from ares.analysis.MultiPlot import MultiPanel

sim = ares.simulations.RaySegment(problem_type=0, optically_thin=1, 
    secondary_ionization=0)
sim.run()

anl = ares.analysis.RaySegment(sim.checkpoints)

t, xHI = anl.CellEvolution(field='h_1')
t, T = anl.CellEvolution(field='Tk')

mp = MultiPanel(dims=(2, 1), panel_size=(1, 0.5))
    
s_per_yr = ares.physics.Constants.s_per_yr
mp.grid[0].loglog(t / s_per_yr, xHI, color = 'k')
mp.grid[1].loglog(t / s_per_yr, T, color = 'k')  

mp.grid[0].set_xlim(1e-6, 1e7)
mp.grid[1].set_xlim(1e-6, 1e7)
mp.grid[0].set_ylim(1e-8, 1.5)
mp.grid[1].set_ylim(1e2, 1e5)
    
mp.grid[0].set_ylabel(r'$x_{\mathrm{HI}}$')
mp.grid[1].set_ylabel(r'$T \ (\mathrm{K})$')
mp.grid[0].set_xlabel(r'$t \ (\mathrm{yr})$')

for ax in mp.grid:
    ax.loglog([sim.pf['source_lifetime'] * 1e6]*2, ax.get_ylim(), 
        color='k', ls=':')

ax.annotate('source OFF', (sim.pf['source_lifetime'] * 1e6, 2e2), ha='right',
    va='bottom', rotation=90)
    
mp.fix_ticks()

