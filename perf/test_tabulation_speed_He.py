"""

test_tabulation_speed_He.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May 12 15:26:39 MDT 2015

Description: How fast can we make look-up tables for Phi and Psi?

"""

import ares
import time

sim = ares.simulations.RaySegment(problem_type=12, tables_discrete_gen=True,
    source_table='bb_He.npz')
#sim.save_tables(prefix='bb_He')

sim.run()

anl = ares.analysis.RaySegment(sim)

ax1 = anl.RadialProfile('h_1', color='k', ls='-', fig=1)
anl.RadialProfile('h_2', color='k', ls='--', ax=ax1)

ax2 = anl.RadialProfile('he_1', color='b', ls='-', fig=2)
anl.RadialProfile('he_2', color='b', ls='--', ax=ax2)
anl.RadialProfile('he_3', color='b', ls=':', ax=ax2)

anl.RadialProfile('Tk', color='b', ls='-', fig=3)

