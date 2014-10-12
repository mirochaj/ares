"""

test_sed_parameterized.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Oct  2 16:57:01 MDT 2013

Description: See if passing spectrum_function parameter works.

"""

import rt1d
import matplotlib.pyplot as pl

BBfunc = rt1d.sources.StellarSource._Planck

# First, use StellarSource class
sim = rt1d.run.Simulation(problem_type=2, stop_time=30)
sim.run()

# Now, use ParameterizedSource class
sim2 = rt1d.run.Simulation(problem_type=2, source_type='parameterized',
    spectrum_function=lambda E: BBfunc(E, 1e5), stop_time=30,
    source_Lbol=sim.rs.src[0].Lbol)
sim2.run()

anl1 = rt1d.analyze.Simulation(sim.checkpoints)
anl2 = rt1d.analyze.Simulation(sim2.checkpoints)

ax = anl1.TemperatureProfile(t=[10, 30])
anl2.TemperatureProfile(t=[10, 30], ax=ax, marker='o', color='b')
