"""

test_rt06_2.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06).

"""

import ares

sim = ares.simulations.RaySegment(problem_type=2)
sim.run()

anl = ares.analysis.RaySegment(sim)

ax2 = anl.RadialProfile('Tk', t=[10, 30, 100])

ax2 = anl.RadialProfile('h_1', t=[10, 30, 100], fig=2)
ax2 = anl.RadialProfile('h_2', t=[10, 30, 100], ax=ax2, ls='--')

