"""

test_multi_phase.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Feb 21 11:55:00 MST 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

sim = ares.simulations.MultiPhaseMedium()
sim.run()

mp = ares.analysis.MultiPanel(dims=(2, 1), panel_size=(1, 0.5))
mp.grid[0].loglog(sim.history['z'], sim.history['igm_Tk'])
mp.grid[1].loglog(sim.history['z'], sim.history['igm_h_2'])
mp.fix_ticks()

