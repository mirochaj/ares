"""

test_tabulation_speed_H.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May 12 15:26:39 MDT 2015

Description: How fast can we make look-up tables for Phi and Psi?

"""

import ares
import time

t1 = time.time()
sim1 = ares.simulations.RaySegment(problem_type=2, tables_discrete_gen=False)
t2 = time.time()

t3 = time.time()
sim2 = ares.simulations.RaySegment(problem_type=2, tables_discrete_gen=True)
t4 = time.time()

print "Discrete tabulation is %.2gx faster than quad." % ((t2 - t1) / (t4 - t3))

sim1.run()
sim2.run()

anl1 = ares.analysis.RaySegment(sim1)
anl2 = ares.analysis.RaySegment(sim2)

ax = anl1.RadialProfile('h_2', color='k')
anl2.RadialProfile('h_2', color='b', ls='--', lw=4, ax=ax)

