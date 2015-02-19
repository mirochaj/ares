"""

test_rt06_1.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: 

"""

import ares

sim = ares.simulations.RaySegment(problem_type=1)
sim.run()

anl = ares.analysis.RaySegment(sim)
ax1 = anl.PlotIonizationFrontEvolution()

ax2 = anl.RadialProfile('h_1', t=[10, 100, 500], fig=2)
ax2 = anl.RadialProfile('h_2', t=[10, 100, 500], ax=ax2, ls='--')
