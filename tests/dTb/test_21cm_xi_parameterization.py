"""

test_21cm_xi_parameterization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Nov 22 14:20:03 MST 2014

Description: 

"""

import ares

sim = ares.simulations.Global21cm()
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()

sim2 = ares.simulations.Global21cm(xi_XR=0.02, xi_LW=969.,
    xi_UV=40.)
sim2.run()    

anl2 = ares.analysis.Global21cm(sim2)
anl2.GlobalSignature(ax=ax, color='b')
