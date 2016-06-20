"""

test_rt06_2.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06), run using three different 
solution methods.

"""

import ares
import matplotlib.pyplot as pl

ax1 = None
ax2 = None
c = ['k', 'b','r']
for i, ptype in enumerate([2, 2.1, 2.2]):
    sim = ares.simulations.RaySegment(problem_type=ptype)
    sim.run()
    
    anl = ares.analysis.RaySegment(sim)
    
    ax1 = anl.RadialProfile('Tk', t=[10, 30, 100], ax=ax1, color=c[i])
    
    ax2 = anl.RadialProfile('h_1', t=[10, 30, 100], ax=ax2, fig=2, color=c[i])
    ax2 = anl.RadialProfile('h_2', t=[10, 30, 100], ax=ax2, ls='--', color=c[i])

pl.draw()
