"""

generate_rc_table.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May 12 20:54:21 MDT 2015

Description: 

"""

import ares

#
## INPUT
helium = 1
dlogN = 0.05
NE = 100
##
#

ptype = 2 + 10 * helium

sim = ares.simulations.RaySegment(problem_type=ptype, 
    tables_discrete_gen=True, tables_energy_bins=NE, tables_dlogN=[dlogN]*3)
sim.save_tables(prefix='bb_He_NE_%i_dlogN_%.2g' % (NE, dlogN))


