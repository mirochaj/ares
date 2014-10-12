"""

test_rt06_1.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: 

"""

import ares
import matplotlib.pyplot as pl

sim = ares.simulations.RaySegment(problem_type=1)
sim.run()

fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

anl = ares.analysis.RaySegment(sim.checkpoints)
anl.PlotIonizationFrontEvolution()

anl.IonizationProfile(t=[10, 100, 500], annotate=True, ax=ax2)



