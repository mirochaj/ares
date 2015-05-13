"""

test_tabulation_speed_He.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May 12 15:26:39 MDT 2015

Description: How fast can we make look-up tables for Phi and Psi?

"""

import ares
import time
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

sim = ares.simulations.RaySegment(problem_type=12, tables_discrete_gen=True,
    source_table='bb_He.npz')
#sim.save_tables(prefix='bb_He')

sim.run()

anl = ares.analysis.RaySegment(sim)

anl.RadialProfile('h_2', color='k', ls='-')
anl.RadialProfile('he_2', color='b', ls='-')
anl.RadialProfile('he_3', color='b', ls='--')

anl.RadialProfile('Tk', color='b', ls='--', fig=2)

