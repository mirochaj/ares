"""

test_21cm_tanh.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep  9 20:03:58 MDT 2014

Description: 

"""

import ares, time

t1 = time.time()
sim = ares.simulations.Global21cm(tanh_model=True)
sim.run()

anl = ares.analysis.Global21cm(sim)
print anl.turning_points

ax = anl.GlobalSignature()

anl.blob_analysis(['z', 'dTb', 'igm_Tk', 'igm_heat'], ['B', 'C', 'D'])

t2 = time.time()

print "Took %.2g seconds" % (t2 - t1)
