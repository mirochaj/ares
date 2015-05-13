"""

test_slab.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun May 10 13:51:54 MDT 2015

Description: 

"""

import ares

sim = ares.simulations.RaySegment(problem_type=3)
sim.run()

anl = ares.analysis.RaySegment(sim)

ax2 = anl.RadialProfile('Tk', t=[10, 30, 100])

ax2 = anl.RadialProfile('h_1', t=[10, 30, 100], fig=2)
ax2 = anl.RadialProfile('h_2', t=[10, 30, 100], ax=ax2, ls='--')



