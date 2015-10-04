"""

test_21cm_basic.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Oct 1 15:23:53 2012

Description: Make sure the global 21-cm signal calculator works.

"""

import ares
import matplotlib.pyplot as pl

sim = ares.simulations.Global21cm()
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()

assert True

