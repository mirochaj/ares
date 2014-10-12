"""

test_rt06_2.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06).

"""

import rt1d
import matplotlib.pyplot as pl

sim = rt1d.run.Simulation(problem_type=2)
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

anl.TemperatureProfile(t=[10, 30, 100], ax=ax1)
anl.IonizationProfile(t=[10, 30, 100], ax=ax2)


